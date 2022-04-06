/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import * as tf from '../index';
import { ALL_ENVS, describeWithFlags } from '../jasmine_util';
import { expectArraysClose } from '../test_util';
describeWithFlags('scatterND', ALL_ENVS, () => {
    it('should work for 2d', async () => {
        const indices = tf.tensor1d([0, 4, 2], 'int32');
        const updates = tf.tensor2d([100, 101, 102, 777, 778, 779, 1000, 1001, 1002], [3, 3], 'int32');
        const shape = [5, 3];
        const result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        expectArraysClose(await result.data(), [100, 101, 102, 0, 0, 0, 1000, 1001, 1002, 0, 0, 0, 777, 778, 779]);
    });
    it('should work for simple 1d', async () => {
        const indices = tf.tensor1d([3], 'int32');
        const updates = tf.tensor1d([101], 'float32');
        const shape = [5];
        const result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        expectArraysClose(await result.data(), [0, 0, 0, 101, 0]);
    });
    it('should work for multiple 1d', async () => {
        const indices = tf.tensor1d([0, 4, 2], 'int32');
        const updates = tf.tensor1d([100, 101, 102], 'float32');
        const shape = [5];
        const result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        expectArraysClose(await result.data(), [100, 0, 102, 0, 101]);
    });
    it('should work for high rank updates', async () => {
        const indices = tf.tensor2d([0, 2], [2, 1], 'int32');
        const updates = tf.tensor3d([
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
        ], [2, 4, 4], 'float32');
        const shape = [4, 4, 4];
        const result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        expectArraysClose(await result.data(), [
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
            8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]);
    });
    it('should work for high rank indices', async () => {
        const indices = tf.tensor2d([0, 2, 0, 1], [2, 2], 'int32');
        const updates = tf.tensor1d([10, 20], 'float32');
        const shape = [3, 3];
        const result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        expectArraysClose(await result.data(), [0, 20, 10, 0, 0, 0, 0, 0, 0]);
    });
    it('should work for high rank indices and update', () => {
        const indices = tf.tensor2d([1, 0, 0, 1, 0, 1], [3, 2], 'int32');
        const updates = tf.ones([3, 256], 'float32');
        const shape = [2, 2, 256];
        const result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
    });
    it('should sum the duplicated indices', async () => {
        const indices = tf.tensor1d([0, 4, 2, 1, 3, 0], 'int32');
        const updates = tf.tensor1d([10, 20, 30, 40, 50, 60], 'float32');
        const shape = [8];
        const result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        expectArraysClose(await result.data(), [70, 40, 30, 50, 20, 0, 0, 0]);
    });
    it('should work for tensorLike input', async () => {
        const indices = [0, 4, 2];
        const updates = [100, 101, 102];
        const shape = [5];
        const result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual('float32');
        expectArraysClose(await result.data(), [100, 0, 102, 0, 101]);
    });
    it('should throw error when indices type is not int32', () => {
        const indices = tf.tensor2d([0, 2, 0, 1], [2, 2], 'float32');
        const updates = tf.tensor1d([10, 20], 'float32');
        const shape = [3, 3];
        expect(() => tf.scatterND(indices, updates, shape)).toThrow();
    });
    it('should throw error when indices and update mismatch', () => {
        const indices = tf.tensor2d([0, 4, 2], [3, 1], 'int32');
        const updates = tf.tensor2d([100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004], [3, 4], 'float32');
        const shape = [5, 3];
        expect(() => tf.scatterND(indices, updates, shape)).toThrow();
    });
    it('should throw error when indices and update count mismatch', () => {
        const indices = tf.tensor2d([0, 4, 2], [3, 1], 'int32');
        const updates = tf.tensor2d([100, 101, 102, 10000, 10001, 10002], [2, 3], 'float32');
        const shape = [5, 3];
        expect(() => tf.scatterND(indices, updates, shape)).toThrow();
    });
    it('should throw error when indices are scalar', () => {
        const indices = tf.scalar(1, 'int32');
        const updates = tf.tensor2d([100, 101, 102, 10000, 10001, 10002], [2, 3], 'float32');
        const shape = [5, 3];
        expect(() => tf.scatterND(indices, updates, shape)).toThrow();
    });
    it('should throw error when update is scalar', () => {
        const indices = tf.tensor2d([0, 4, 2], [3, 1], 'int32');
        const updates = tf.scalar(1, 'float32');
        const shape = [5, 3];
        expect(() => tf.scatterND(indices, updates, shape)).toThrow();
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2NhdHRlcl9uZF90ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvc2NhdHRlcl9uZF90ZXN0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sS0FBSyxFQUFFLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxRQUFRLEVBQUUsaUJBQWlCLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUM1RCxPQUFPLEVBQUMsaUJBQWlCLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFFL0MsaUJBQWlCLENBQUMsV0FBVyxFQUFFLFFBQVEsRUFBRSxHQUFHLEVBQUU7SUFDNUMsRUFBRSxDQUFDLG9CQUFvQixFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ2xDLE1BQU0sT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQ2hELE1BQU0sT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQ3ZCLENBQUMsR0FBRyxFQUFFLEdBQUcsRUFBRSxHQUFHLEVBQUUsR0FBRyxFQUFFLEdBQUcsRUFBRSxHQUFHLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztRQUN2RSxNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNyQixNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLE9BQU8sRUFBRSxPQUFPLEVBQUUsS0FBSyxDQUFDLENBQUM7UUFDckQsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDcEMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQzVDLGlCQUFpQixDQUNiLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUNuQixDQUFDLEdBQUcsRUFBRSxHQUFHLEVBQUUsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsSUFBSSxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLEdBQUcsRUFBRSxHQUFHLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUMxRSxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQywyQkFBMkIsRUFBRSxLQUFLLElBQUksRUFBRTtRQUN6QyxNQUFNLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFDMUMsTUFBTSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLFNBQVMsQ0FBQyxDQUFDO1FBQzlDLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEIsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDLFNBQVMsQ0FBQyxPQUFPLEVBQUUsT0FBTyxFQUFFLEtBQUssQ0FBQyxDQUFDO1FBQ3JELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ3BDLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUM1QyxpQkFBaUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzVELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDZCQUE2QixFQUFFLEtBQUssSUFBSSxFQUFFO1FBQzNDLE1BQU0sT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQ2hELE1BQU0sT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLEVBQUUsR0FBRyxFQUFFLEdBQUcsQ0FBQyxFQUFFLFNBQVMsQ0FBQyxDQUFDO1FBQ3hELE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEIsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDLFNBQVMsQ0FBQyxPQUFPLEVBQUUsT0FBTyxFQUFFLEtBQUssQ0FBQyxDQUFDO1FBQ3JELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ3BDLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUM1QyxpQkFBaUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQyxDQUFDO0lBQ2hFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLG1DQUFtQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ2pELE1BQU0sT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFDckQsTUFBTSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FDdkI7WUFDRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQzlDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUM7U0FDL0MsRUFDRCxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsU0FBUyxDQUFDLENBQUM7UUFDMUIsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxTQUFTLENBQUMsT0FBTyxFQUFFLE9BQU8sRUFBRSxLQUFLLENBQUMsQ0FBQztRQUNyRCxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUNwQyxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDNUMsaUJBQWlCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUU7WUFDckMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUNoRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQ2hFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDO1NBQzNELENBQUMsQ0FBQztJQUNMLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLG1DQUFtQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ2pELE1BQU0sT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztRQUMzRCxNQUFNLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLFNBQVMsQ0FBQyxDQUFDO1FBQ2pELE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JCLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxTQUFTLENBQUMsT0FBTyxFQUFFLE9BQU8sRUFBRSxLQUFLLENBQUMsQ0FBQztRQUNyRCxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUNwQyxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxP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QyxNQUFNLENBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQ3RDLE1BQU0sT0FBTyxHQUNULEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLEVBQUUsR0FBRyxFQUFFLEdBQUcsRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUFFLEtBQUssQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLFNBQVMsQ0FBQyxDQUFDO1FBQ3pFLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsU0FBUyxDQUFDLE9BQU8sRUFBRSxPQUFPLEVBQUUsS0FBSyxDQUFDLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztJQUNoRSxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQywwQ0FBMEMsRUFBRSxHQUFHLEVBQUU7UUFDbEQsTUFBTSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFDeEQsTUFBTSxPQUFPLEdBQUcsRUFBRSxDQUFDLE1BQU0sQ0FBQyxDQUFDLEVBQUUsU0FBUyxDQUFDLENBQUM7UUFDeEMsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxTQUFTLENBQUMsT0FBTyxFQUFFLE9BQU8sRUFBRSxLQUFLLENBQUMsQ0FBQyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ2hFLENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCAqIGFzIHRmIGZyb20gJy4uL2luZGV4JztcbmltcG9ydCB7QUxMX0VOVlMsIGRlc2NyaWJlV2l0aEZsYWdzfSBmcm9tICcuLi9qYXNtaW5lX3V0aWwnO1xuaW1wb3J0IHtleHBlY3RBcnJheXNDbG9zZX0gZnJvbSAnLi4vdGVzdF91dGlsJztcblxuZGVzY3JpYmVXaXRoRmxhZ3MoJ3NjYXR0ZXJORCcsIEFMTF9FTlZTLCAoKSA9PiB7XG4gIGl0KCdzaG91bGQgd29yayBmb3IgMmQnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5kaWNlcyA9IHRmLnRlbnNvcjFkKFswLCA0LCAyXSwgJ2ludDMyJyk7XG4gICAgY29uc3QgdXBkYXRlcyA9IHRmLnRlbnNvcjJkKFxuICAgICAgICBbMTAwLCAxMDEsIDEwMiwgNzc3LCA3NzgsIDc3OSwgMTAwMCwgMTAwMSwgMTAwMl0sIFszLCAzXSwgJ2ludDMyJyk7XG4gICAgY29uc3Qgc2hhcGUgPSBbNSwgM107XG4gICAgY29uc3QgcmVzdWx0ID0gdGYuc2NhdHRlck5EKGluZGljZXMsIHVwZGF0ZXMsIHNoYXBlKTtcbiAgICBleHBlY3QocmVzdWx0LnNoYXBlKS50b0VxdWFsKHNoYXBlKTtcbiAgICBleHBlY3QocmVzdWx0LmR0eXBlKS50b0VxdWFsKHVwZGF0ZXMuZHR5cGUpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKFxuICAgICAgICBhd2FpdCByZXN1bHQuZGF0YSgpLFxuICAgICAgICBbMTAwLCAxMDEsIDEwMiwgMCwgMCwgMCwgMTAwMCwgMTAwMSwgMTAwMiwgMCwgMCwgMCwgNzc3LCA3NzgsIDc3OV0pO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIHdvcmsgZm9yIHNpbXBsZSAxZCcsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBpbmRpY2VzID0gdGYudGVuc29yMWQoWzNdLCAnaW50MzInKTtcbiAgICBjb25zdCB1cGRhdGVzID0gdGYudGVuc29yMWQoWzEwMV0sICdmbG9hdDMyJyk7XG4gICAgY29uc3Qgc2hhcGUgPSBbNV07XG4gICAgY29uc3QgcmVzdWx0ID0gdGYuc2NhdHRlck5EKGluZGljZXMsIHVwZGF0ZXMsIHNoYXBlKTtcbiAgICBleHBlY3QocmVzdWx0LnNoYXBlKS50b0VxdWFsKHNoYXBlKTtcbiAgICBleHBlY3QocmVzdWx0LmR0eXBlKS50b0VxdWFsKHVwZGF0ZXMuZHR5cGUpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIFswLCAwLCAwLCAxMDEsIDBdKTtcbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCB3b3JrIGZvciBtdWx0aXBsZSAxZCcsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBpbmRpY2VzID0gdGYudGVuc29yMWQoWzAsIDQsIDJdLCAnaW50MzInKTtcbiAgICBjb25zdCB1cGRhdGVzID0gdGYudGVuc29yMWQoWzEwMCwgMTAxLCAxMDJdLCAnZmxvYXQzMicpO1xuICAgIGNvbnN0IHNoYXBlID0gWzVdO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRmLnNjYXR0ZXJORChpbmRpY2VzLCB1cGRhdGVzLCBzaGFwZSk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChzaGFwZSk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9FcXVhbCh1cGRhdGVzLmR0eXBlKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBbMTAwLCAwLCAxMDIsIDAsIDEwMV0pO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIHdvcmsgZm9yIGhpZ2ggcmFuayB1cGRhdGVzJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGluZGljZXMgPSB0Zi50ZW5zb3IyZChbMCwgMl0sIFsyLCAxXSwgJ2ludDMyJyk7XG4gICAgY29uc3QgdXBkYXRlcyA9IHRmLnRlbnNvcjNkKFxuICAgICAgICBbXG4gICAgICAgICAgNSwgNSwgNSwgNSwgNiwgNiwgNiwgNiwgNywgNywgNywgNywgOCwgOCwgOCwgOCxcbiAgICAgICAgICA1LCA1LCA1LCA1LCA2LCA2LCA2LCA2LCA3LCA3LCA3LCA3LCA4LCA4LCA4LCA4XG4gICAgICAgIF0sXG4gICAgICAgIFsyLCA0LCA0XSwgJ2Zsb2F0MzInKTtcbiAgICBjb25zdCBzaGFwZSA9IFs0LCA0LCA0XTtcbiAgICBjb25zdCByZXN1bHQgPSB0Zi5zY2F0dGVyTkQoaW5kaWNlcywgdXBkYXRlcywgc2hhcGUpO1xuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoc2hhcGUpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvRXF1YWwodXBkYXRlcy5kdHlwZSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgW1xuICAgICAgNSwgNSwgNSwgNSwgNiwgNiwgNiwgNiwgNywgNywgNywgNywgOCwgOCwgOCwgOCwgMCwgMCwgMCwgMCwgMCwgMCxcbiAgICAgIDAsIDAsIDAsIDAsIDAsIDAsIDAsIDAsIDAsIDAsIDUsIDUsIDUsIDUsIDYsIDYsIDYsIDYsIDcsIDcsIDcsIDcsXG4gICAgICA4LCA4LCA4LCA4LCAwLCAwLCAwLCAwLCAwLCAwLCAwLCAwLCAwLCAwLCAwLCAwLCAwLCAwLCAwLCAwXG4gICAgXSk7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgd29yayBmb3IgaGlnaCByYW5rIGluZGljZXMnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5kaWNlcyA9IHRmLnRlbnNvcjJkKFswLCAyLCAwLCAxXSwgWzIsIDJdLCAnaW50MzInKTtcbiAgICBjb25zdCB1cGRhdGVzID0gdGYudGVuc29yMWQoWzEwLCAyMF0sICdmbG9hdDMyJyk7XG4gICAgY29uc3Qgc2hhcGUgPSBbMywgM107XG4gICAgY29uc3QgcmVzdWx0ID0gdGYuc2NhdHRlck5EKGluZGljZXMsIHVwZGF0ZXMsIHNoYXBlKTtcbiAgICBleHBlY3QocmVzdWx0LnNoYXBlKS50b0VxdWFsKHNoYXBlKTtcbiAgICBleHBlY3QocmVzdWx0LmR0eXBlKS50b0VxdWFsKHVwZGF0ZXMuZHR5cGUpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIFswLCAyMCwgMTAsIDAsIDAsIDAsIDAsIDAsIDBdKTtcbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCB3b3JrIGZvciBoaWdoIHJhbmsgaW5kaWNlcyBhbmQgdXBkYXRlJywgKCkgPT4ge1xuICAgIGNvbnN0IGluZGljZXMgPSB0Zi50ZW5zb3IyZChbMSwgMCwgMCwgMSwgMCwgMV0sIFszLCAyXSwgJ2ludDMyJyk7XG4gICAgY29uc3QgdXBkYXRlcyA9IHRmLm9uZXMoWzMsIDI1Nl0sICdmbG9hdDMyJyk7XG4gICAgY29uc3Qgc2hhcGUgPSBbMiwgMiwgMjU2XTtcbiAgICBjb25zdCByZXN1bHQgPSB0Zi5zY2F0dGVyTkQoaW5kaWNlcywgdXBkYXRlcywgc2hhcGUpO1xuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoc2hhcGUpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvRXF1YWwodXBkYXRlcy5kdHlwZSk7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgc3VtIHRoZSBkdXBsaWNhdGVkIGluZGljZXMnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5kaWNlcyA9IHRmLnRlbnNvcjFkKFswLCA0LCAyLCAxLCAzLCAwXSwgJ2ludDMyJyk7XG4gICAgY29uc3QgdXBkYXRlcyA9IHRmLnRlbnNvcjFkKFsxMCwgMjAsIDMwLCA0MCwgNTAsIDYwXSwgJ2Zsb2F0MzInKTtcbiAgICBjb25zdCBzaGFwZSA9IFs4XTtcbiAgICBjb25zdCByZXN1bHQgPSB0Zi5zY2F0dGVyTkQoaW5kaWNlcywgdXBkYXRlcywgc2hhcGUpO1xuICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoc2hhcGUpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvRXF1YWwodXBkYXRlcy5kdHlwZSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgWzcwLCA0MCwgMzAsIDUwLCAyMCwgMCwgMCwgMF0pO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIHdvcmsgZm9yIHRlbnNvckxpa2UgaW5wdXQnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgaW5kaWNlcyA9IFswLCA0LCAyXTtcbiAgICBjb25zdCB1cGRhdGVzID0gWzEwMCwgMTAxLCAxMDJdO1xuICAgIGNvbnN0IHNoYXBlID0gWzVdO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRmLnNjYXR0ZXJORChpbmRpY2VzLCB1cGRhdGVzLCBzaGFwZSk7XG4gICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChzaGFwZSk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9FcXVhbCgnZmxvYXQzMicpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIFsxMDAsIDAsIDEwMiwgMCwgMTAxXSk7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgdGhyb3cgZXJyb3Igd2hlbiBpbmRpY2VzIHR5cGUgaXMgbm90IGludDMyJywgKCkgPT4ge1xuICAgIGNvbnN0IGluZGljZXMgPSB0Zi50ZW5zb3IyZChbMCwgMiwgMCwgMV0sIFsyLCAyXSwgJ2Zsb2F0MzInKTtcbiAgICBjb25zdCB1cGRhdGVzID0gdGYudGVuc29yMWQoWzEwLCAyMF0sICdmbG9hdDMyJyk7XG4gICAgY29uc3Qgc2hhcGUgPSBbMywgM107XG4gICAgZXhwZWN0KCgpID0+IHRmLnNjYXR0ZXJORChpbmRpY2VzLCB1cGRhdGVzLCBzaGFwZSkpLnRvVGhyb3coKTtcbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCB0aHJvdyBlcnJvciB3aGVuIGluZGljZXMgYW5kIHVwZGF0ZSBtaXNtYXRjaCcsICgpID0+IHtcbiAgICBjb25zdCBpbmRpY2VzID0gdGYudGVuc29yMmQoWzAsIDQsIDJdLCBbMywgMV0sICdpbnQzMicpO1xuICAgIGNvbnN0IHVwZGF0ZXMgPSB0Zi50ZW5zb3IyZChcbiAgICAgICAgWzEwMCwgMTAxLCAxMDIsIDEwMywgNzc3LCA3NzgsIDc3OSwgNzgwLCAxMDAwMCwgMTAwMDEsIDEwMDAyLCAxMDAwNF0sXG4gICAgICAgIFszLCA0XSwgJ2Zsb2F0MzInKTtcbiAgICBjb25zdCBzaGFwZSA9IFs1LCAzXTtcbiAgICBleHBlY3QoKCkgPT4gdGYuc2NhdHRlck5EKGluZGljZXMsIHVwZGF0ZXMsIHNoYXBlKSkudG9UaHJvdygpO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIHRocm93IGVycm9yIHdoZW4gaW5kaWNlcyBhbmQgdXBkYXRlIGNvdW50IG1pc21hdGNoJywgKCkgPT4ge1xuICAgIGNvbnN0IGluZGljZXMgPSB0Zi50ZW5zb3IyZChbMCwgNCwgMl0sIFszLCAxXSwgJ2ludDMyJyk7XG4gICAgY29uc3QgdXBkYXRlcyA9XG4gICAgICAgIHRmLnRlbnNvcjJkKFsxMDAsIDEwMSwgMTAyLCAxMDAwMCwgMTAwMDEsIDEwMDAyXSwgWzIsIDNdLCAnZmxvYXQzMicpO1xuICAgIGNvbnN0IHNoYXBlID0gWzUsIDNdO1xuICAgIGV4cGVjdCgoKSA9PiB0Zi5zY2F0dGVyTkQoaW5kaWNlcywgdXBkYXRlcywgc2hhcGUpKS50b1Rocm93KCk7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgdGhyb3cgZXJyb3Igd2hlbiBpbmRpY2VzIGFyZSBzY2FsYXInLCAoKSA9PiB7XG4gICAgY29uc3QgaW5kaWNlcyA9IHRmLnNjYWxhcigxLCAnaW50MzInKTtcbiAgICBjb25zdCB1cGRhdGVzID1cbiAgICAgICAgdGYudGVuc29yMmQoWzEwMCwgMTAxLCAxMDIsIDEwMDAwLCAxMDAwMSwgMTAwMDJdLCBbMiwgM10sICdmbG9hdDMyJyk7XG4gICAgY29uc3Qgc2hhcGUgPSBbNSwgM107XG4gICAgZXhwZWN0KCgpID0+IHRmLnNjYXR0ZXJORChpbmRpY2VzLCB1cGRhdGVzLCBzaGFwZSkpLnRvVGhyb3coKTtcbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCB0aHJvdyBlcnJvciB3aGVuIHVwZGF0ZSBpcyBzY2FsYXInLCAoKSA9PiB7XG4gICAgY29uc3QgaW5kaWNlcyA9IHRmLnRlbnNvcjJkKFswLCA0LCAyXSwgWzMsIDFdLCAnaW50MzInKTtcbiAgICBjb25zdCB1cGRhdGVzID0gdGYuc2NhbGFyKDEsICdmbG9hdDMyJyk7XG4gICAgY29uc3Qgc2hhcGUgPSBbNSwgM107XG4gICAgZXhwZWN0KCgpID0+IHRmLnNjYXR0ZXJORChpbmRpY2VzLCB1cGRhdGVzLCBzaGFwZSkpLnRvVGhyb3coKTtcbiAgfSk7XG59KTtcbiJdfQ==