/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/resize_nearest_neighbor" />
import { Tensor3D, Tensor4D } from '../../tensor';
import { TensorLike } from '../../types';
/**
 * NearestNeighbor resize a batch of 3D images to a new shape.
 *
 * @param images The images, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param size The new shape `[newHeight, newWidth]` to resize the
 *     images to. Each channel is resized individually.
 * @param alignCorners Defaults to False. If true, rescale
 *     input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
 *     corners of images and resized images. If false, rescale by
 *     `new_height / height`. Treat similarly the width dimension.
 * @param halfPixelCenters Defaults to `false`. Whether to assumes pixels are of
 *      half the actual dimensions, and yields more accurate resizes. This flag
 *      would also make the floating point coordinates of the top left pixel
 *      0.5, 0.5.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function resizeNearestNeighbor_<T extends Tensor3D | Tensor4D>(images: T | TensorLike, size: [number, number], alignCorners?: boolean, halfPixelCenters?: boolean): T;
export declare const resizeNearestNeighbor: typeof resizeNearestNeighbor_;
export {};
