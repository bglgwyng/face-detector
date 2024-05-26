import { Fragment, useEffect, useRef, useState } from "react";
import "./App.css";
import { type Detection, Detector } from "./detector";
import { SSD_OPTIONS_BACK, getTensorFromImageUrl } from "./utils";

const { tf, tflite } = window;

const frontModel = await tflite.loadTFLiteModel(
	"./face_detection_front.tflite",
);

const imageUrl = "./newjeans.jpg";
const backModel = await tflite.loadTFLiteModel("./face_detection_back.tflite");
// console.info(window)
function App() {
	const [boxes, setBoxes] = useState<Detection[]>([]);

	const canvasRef = useRef<HTMLCanvasElement>(null);

	useEffect(() => {
		const image = new Image();
		image.src = imageUrl;
		image.onload = async () => {
			const detector = new Detector(backModel, SSD_OPTIONS_BACK);
			const { inputShape } = detector;

			const tensor = await getTensorFromImageUrl(
				imageUrl,
				inputShape.width,
				inputShape.height,
				canvasRef.current!,
			);

			const faces = await detector.detect(tensor);
			setBoxes(faces);
		};
	}, []);

	return (
		<div>
			<canvas ref={canvasRef} style={{ width: 500, height: 500 }} />
			<div
				style={{
					display: "flex",
					position: "relative",
					width: 500,
					height: 500,
					backgroundColor: "black",
				}}
			>
				<img
					alt=""
					src={imageUrl}
					style={{
						position: "absolute",
						width: 500,
						height: 500,
						objectFit: "contain",
					}}
				/>
				{boxes.map(({ bbox: { xmax, xmin, ymax, ymin }, keypoints }, i) => (
					<Fragment key={i}>
						<div
							style={{
								position: "absolute",
								border: "1px solid blue",
								left: `${xmin * 100}%`,
								top: `${ymin * 100}%`,
								width: `${(xmax - xmin) * 100}%`,
								height: `${(ymax - ymin) * 100}%`,
							}}
						/>
						{keypoints.map(([x, y], i) => (
							<div
								key={i}
								style={{
									position: "absolute",
									border: "1px solid blue",
									left: `${x * 100}%`,
									top: `${y * 100}%`,
									width: 5,
									height: 5,
								}}
							/>
						))}
					</Fragment>
				))}
			</div>
		</div>
	);
}

export default App;
