export {};

import type * as tfTypes from "@tensorflow/tfjs-core/dist/index"
import type * as tfLiteTypes from "@tensorflow/tfjs-tflite/dist/index"
type tfLiteType = typeof tfLiteTypes
type tfType = typeof tfTypes

declare global {
    interface Window { 
        tf: tfType; 
        tflite: tfLiteType; 
    }
}