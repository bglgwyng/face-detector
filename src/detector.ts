import type { Tensor } from "@tensorflow/tfjs-core";
import {
	decodeBoxes,
	generateAnchors,
	RAW_SCORE_LIMIT,
	sigmoid,
	SSD_OPTIONS_BACK,
	SSD_OPTIONS_FRONT,
	type SSD_OPTS,
} from "./utils";
import { clamp, zip } from "ramda";
import assert from "assert";
import type { TFLiteModel } from "@tensorflow/tfjs-tflite";

const { tf, tflite } = window;

export class Detector {
	anchor: [number, number][];
	constructor(
		private model: TFLiteModel,
		ssdOptions: SSD_OPTS,
	) {
		this.anchor = generateAnchors(ssdOptions);
		console.info(model.inputs[0].shape);
	}

	get inputShape() {
		const [, height, width] = this.model.inputs[0].shape!;
		return { width, height };
	}

	detect = async (image: Tensor) => {
		const { width } = this.inputShape;
		// assert(width === 256)
		image = tf.slice(image, [0, 0, 0], [-1, -1, 3]);
		image = tf.div(tf.sub(image, 128), 128);
		image = tf.cast(image, "float32");
		image = tf.expandDims(image, 0);
		// console.info(JSON.stringify(tensor.arraySync()));
		const { regressors, classificators } = this.model.predict(image) as Record<
			string,
			Tensor
		>;

		// console.info("?", model.outputs, await regressors.array());
		// console.info(anchor);
		const rawBoxes = (await tf.squeeze(regressors, [0]).array()) as number[][];
		// console.info(rawBoxes[0].length, rawBoxes[0][0].length);
		const boxes$ = decodeBoxes(width, this.anchor, rawBoxes);
		const scores = tf.squeeze(classificators, [0, 2]).arraySync() as number[];

		const clamp$ = clamp(-RAW_SCORE_LIMIT, RAW_SCORE_LIMIT);
		const scores$ = scores.map((x) => sigmoid(clamp$(x)));
		const boxes$$ = zip(boxes$, scores$).filter(([box, score]) => score >= 0.3);

		const detections = boxes$$.map(
			([box, score]): Detection => new Detection(score, box),
		);

		return nonMaximumSuppression(detections, 0.5, 0.3, true);
	};
}

export class Detection {
	constructor(
		public score: number,
		public data: number[],
	) {
		this.score = score;
		this.data = data;
	}

	get bbox(): BBox {
		return {
			xmin: this.data[0],
			ymin: this.data[1],
			xmax: this.data[2],
			ymax: this.data[3],
		};
	}

	get keypoints(): [number, number][] {
		const keypoints: [number, number][] = [];
		for (let i = 4; i < this.data.length; i += 2) {
			keypoints.push([this.data[i], this.data[i + 1]]);
		}
		return keypoints;
	}
}

export interface BBox {
	xmin: number;
	ymin: number;
	xmax: number;
	ymax: number;
}

function nonMaximumSuppression(
	detections: Detection[],
	minSuppressionThreshold: number,
	minScore?: number,
	weighted = false,
): Detection[] {
	const scores = detections.map((detection) => detection.score);
	const indexedScores = scores.map((score, index) => [index, score] as const);
	indexedScores.sort((a, b) => b[1] - a[1]);

	if (weighted) {
		return weightedNonMaximumSuppression(
			indexedScores,
			detections,
			minSuppressionThreshold,
			minScore,
		);
	}
	return nonMaximumSuppression$(
		indexedScores,
		detections,
		minSuppressionThreshold,
		minScore,
	);
}

function overlapSimilarity(box1: BBox, box2: BBox): number {
	const intersection = intersect(box1, box2);
	if (intersection === null) {
		return 0;
	}
	const intersectArea = intersection.area;
	const denominator = area(box1) + area(box2) - intersectArea;
	return denominator > 0 ? intersectArea / denominator : 0;
}

function nonMaximumSuppression$(
	indexedScores: (readonly [number, number])[],
	detections: Detection[],
	minSuppressionThreshold: number,
	minScore?: number,
): Detection[] {
	const keptBoxes: BBox[] = [];
	const outputs: Detection[] = [];

	for (const [index, score] of indexedScores) {
		if (minScore !== undefined && score < minScore) {
			break;
		}
		const detection = detections[index];
		const bbox = detection.bbox;
		let suppressed = false;

		for (const kept of keptBoxes) {
			const similarity = overlapSimilarity(kept, bbox);
			if (similarity > minSuppressionThreshold) {
				suppressed = true;
				break;
			}
		}

		if (!suppressed) {
			outputs.push(detection);
			keptBoxes.push(bbox);
		}
	}

	return outputs;
}

function weightedNonMaximumSuppression(
	indexedScores: (readonly [number, number])[],
	detections: Detection[],
	minSuppressionThreshold: number,
	minScore?: number,
): Detection[] {
	const remainingIndexedScores = [...indexedScores];
	const outputs: Detection[] = [];

	while (remainingIndexedScores.length > 0) {
		const detection = detections[remainingIndexedScores[0][0]];
		if (minScore !== undefined && detection.score < minScore) {
			break;
		}
		const numPrevIndexedScores = remainingIndexedScores.length;
		const detectionBbox = detection.bbox;

		const remaining: [number, number][] = [];
		const candidates: [number, number][] = [];

		let weightedDetection = detection;

		for (const [index, score] of remainingIndexedScores) {
			const remainingBbox = detections[index].bbox;
			const similarity = overlapSimilarity(remainingBbox, detectionBbox);
			if (similarity > minSuppressionThreshold) {
				candidates.push([index, score]);
			} else {
				remaining.push([index, score]);
			}
		}

		if (candidates.length > 0) {
			const weighted: number[] = Array.from({ length: 16 }, () => 0);
			let totalScore = 0;

			for (const [index, score] of candidates) {
				totalScore += score;
				const data = detections[index].data;
				for (let i = 0; i < data.length; i++) {
					weighted[i] += data[i] * score;
				}
			}

			for (let i = 0; i < weighted.length; i++) {
				weighted[i] /= totalScore;
			}

			// console.info(weighted)
			weightedDetection = new Detection(detection.score, weighted);
		}

		outputs.push(weightedDetection);

		if (numPrevIndexedScores === remaining.length) {
			break;
		}

		remainingIndexedScores.length = 0;
		remainingIndexedScores.push(...remaining);
	}

	return outputs;
}

function intersect(x: BBox, y: BBox) {
	const xmin = Math.max(x.xmin, y.xmin);
	const ymin = Math.max(x.ymin, y.ymin);
	const xmax = Math.min(x.xmax, y.xmax);
	const ymax = Math.min(x.ymax, y.ymax);
	return xmin < xmax && ymin < ymax
		? { xmin, ymin, xmax, ymax, area: (xmax - xmin) * (ymax - ymin) }
		: null;
}

function area(box: BBox) {
	return (box.xmax - box.xmin) * (box.ymax - box.ymin);
}

export const rightEyeIndex = 0;
export const leftEyeIndex = 1;
export const noseIndex = 2;
export const mouthIndex = 3;
export const rightCheekIndex = 4;
export const leftCheekIndex = 5;
