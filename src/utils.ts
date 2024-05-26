import { clamp } from "ramda";

const { tf } = window;

export function generateAnchors(opts: SSD_OPTS): [number, number][] {
	const {
		num_layers,
		strides,
		input_size_height,
		input_size_width,
		anchor_offset_x,
		anchor_offset_y,
		interpolated_scale_aspect_ratio,
	} = opts;

	const anchors: [number, number][] = [];
	let layer_id = 0;

	while (layer_id < num_layers) {
		let last_same_stride_layer = layer_id;
		let repeats = 0;

		while (
			last_same_stride_layer < num_layers &&
			strides[last_same_stride_layer] === strides[layer_id]
		) {
			last_same_stride_layer++;
			repeats += 2;
			if (interpolated_scale_aspect_ratio !== 1.0) {
				repeats++;
			}
		}

		const stride = strides[layer_id];
		const feature_map_height = input_size_height / stride;
		const feature_map_width = input_size_width / stride;

		for (let y = 0; y < feature_map_height; y++) {
			const y_center = (y + anchor_offset_y) / feature_map_height;

			for (let x = 0; x < feature_map_width; x++) {
				const x_center = (x + anchor_offset_x) / feature_map_width;

				for (let i = 0; i < repeats; i++) {
					anchors.push([x_center, y_center]);
				}
			}
		}

		layer_id = last_same_stride_layer;
	}

	return anchors;
}

export function decodeBoxes(
	size: number,
	anchors: [number, number][],
	raw_boxes: number[][],
): number[][] {
	return raw_boxes.map((box, i) => {
		const [anchorX, anchorY] = anchors[i];
		const [x_center, y_center, w, h, ...keypoints] = box;

		const halfW = w / size / 2;
		const halfH = h / size / 2;

		const centerX = x_center / size + anchorX;
		const centerY = y_center / size + anchorY;

		return [
			centerX - halfW,
			centerY - halfH,
			centerX + halfW,
			centerY + halfH,
			...keypoints.map((x, i) => x / size + (i % 2 === 0 ? anchorX : anchorY)),
		];
	});
}

export const SSD_OPTIONS_FRONT: SSD_OPTS = {
	num_layers: 4,
	input_size_height: 128,
	input_size_width: 128,
	anchor_offset_x: 0.5,
	anchor_offset_y: 0.5,
	strides: [8, 16, 16, 16],
	interpolated_scale_aspect_ratio: 1.0,
};

export const SSD_OPTIONS_BACK = {
	num_layers: 4,
	input_size_height: 256,
	input_size_width: 256,
	anchor_offset_x: 0.5,
	anchor_offset_y: 0.5,
	strides: [16, 32, 32, 32],
	interpolated_scale_aspect_ratio: 1.0,
};

export type SSD_OPTS = {
	num_layers: number;
	strides: number[];
	input_size_height: number;
	input_size_width: number;
	anchor_offset_x: number;
	anchor_offset_y: number;
	interpolated_scale_aspect_ratio: number;
};

// console.info(JSON.stringify(generateAnchors(SSD_OPTIONS_FRONT)));
export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export const RAW_SCORE_LIMIT = 80;

export async function getTensorFromImageUrl(
	url: string,
	width: number,
	height: number,
	canvas?: HTMLCanvasElement,
) {
	const inputShape = { width, height };
	return new Promise((resolve, reject) => {
		const image = new Image();
		image.src = url;
		image.onload = async () => {
			canvas ??= document.createElement("canvas");
			canvas.width = inputShape.width;
			canvas.height = inputShape.height;
			const context = canvas.getContext("2d")!;

			const aspectRatio = image.width / image.height;
			const canvasAspectRatio = inputShape.width / inputShape.height;
			const width =
				aspectRatio > canvasAspectRatio
					? inputShape.width
					: inputShape.height * aspectRatio;
			const height =
				aspectRatio > canvasAspectRatio
					? inputShape.width / aspectRatio
					: inputShape.height;

			// fill black
			context.fillStyle = "black";
			context.fillRect(0, 0, inputShape.width, inputShape.height);
			context.drawImage(
				image,
				(inputShape.width - width) / 2,
				(inputShape.height - height) / 2,
				width,
				height,
			);
			const imageData = context.getImageData(
				0,
				0,
				inputShape.width,
				inputShape.height,
			);

			resolve(tf.browser.fromPixels(imageData));
		};
		image.onerror = reject;
	});
}
