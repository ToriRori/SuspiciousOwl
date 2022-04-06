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
import * as tf from '../index';
import { ALL_ENVS, describeWithFlags } from '../jasmine_util';
import { expectArraysClose, expectArraysEqual } from '../test_util';
describeWithFlags('prod', ALL_ENVS, () => {
    it('basic', async () => {
        const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
        const result = tf.prod(a);
        expectArraysClose(await result.data(), 0);
    });
    it('propagates NaNs', async () => {
        const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
        expectArraysEqual(await tf.prod(a).data(), NaN);
    });
    it('prod over dtype int32', async () => {
        const a = tf.tensor1d([1, 5, 7, 3], 'int32');
        const prod = tf.prod(a);
        expectArraysEqual(await prod.data(), 105);
    });
    it('prod over dtype bool', async () => {
        const a = tf.tensor1d([true, false, false, true, true], 'bool');
        const prod = tf.prod(a);
        expectArraysEqual(await prod.data(), 0);
    });
    it('prods all values in 2D array with keep dim', async () => {
        const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
        const res = tf.prod(a, null, true /* keepDims */);
        expect(res.shape).toEqual([1, 1]);
        expectArraysClose(await res.data(), 0);
    });
    it('prods across axis=0 in 2D array', async () => {
        const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
        const res = tf.prod(a, [0]);
        expect(res.shape).toEqual([2]);
        expectArraysClose(await res.data(), [0, 2]);
    });
    it('prods across axis=0 in 2D array, keepDims', async () => {
        const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
        const res = tf.prod(a, [0], true /* keepDims */);
        expect(res.shape).toEqual([1, 2]);
        expectArraysClose(await res.data(), [0, 2]);
    });
    it('prods across axis=1 in 2D array', async () => {
        const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
        const res = tf.prod(a, [1]);
        expect(res.shape).toEqual([3]);
        expectArraysClose(await res.data(), [2, 3, 1]);
    });
    it('2D, axis=1 provided as number', async () => {
        const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [2, 3]);
        const res = tf.prod(a, 1);
        expect(res.shape).toEqual([2]);
        expectArraysClose(await res.data(), [6, 1]);
    });
    it('2D, axis = -1 provided as number', async () => {
        const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [2, 3]);
        const res = tf.prod(a, -1);
        expect(res.shape).toEqual([2]);
        expectArraysClose(await res.data(), [6, 1]);
    });
    it('prods across axis=0,1 in 2D array', async () => {
        const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
        const res = tf.prod(a, [0, 1]);
        expect(res.shape).toEqual([]);
        expectArraysClose(await res.data(), [6]);
    });
    it('2D, axis=[-1,-2] in 2D array', async () => {
        const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
        const res = tf.prod(a, [-1, -2]);
        expect(res.shape).toEqual([]);
        expectArraysClose(await res.data(), [6]);
    });
    it('throws when passed a non-tensor', () => {
        expect(() => tf.prod({}))
            .toThrowError(/Argument 'x' passed to 'prod' must be a Tensor/);
    });
    it('accepts a tensor-like object', async () => {
        const result = tf.prod([[1, 2], [3, 1], [1, 1]]);
        expectArraysClose(await result.data(), 6);
    });
    it('throws error for string tensor', () => {
        expect(() => tf.prod(['a']))
            .toThrowError(/Argument 'x' passed to 'prod' must be numeric tensor/);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicHJvZF90ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvcHJvZF90ZXN0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sS0FBSyxFQUFFLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxRQUFRLEVBQUUsaUJBQWlCLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUM1RCxPQUFPLEVBQUMsaUJBQWlCLEVBQUUsaUJBQWlCLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFFbEUsaUJBQWlCLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxHQUFHLEVBQUU7SUFDdkMsRUFBRSxDQUFDLE9BQU8sRUFBRSxLQUFLLElBQUksRUFBRTtRQUNyQixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2xELE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDMUIsaUJBQWlCLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDNUMsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsaUJBQWlCLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDL0IsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNwRCxpQkFBaUIsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7SUFDbEQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsdUJBQXVCLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDckMsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQzdDLE1BQU0sSUFBSSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDeEIsaUJBQWlCLENBQUMsTUFBTSxJQUFJLENBQUMsSUFBSSxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7SUFDNUMsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsc0JBQXNCLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDcEMsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsRUFBRSxNQUFNLENBQUMsQ0FBQztRQUNoRSxNQUFNLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLGlCQUFpQixDQUFDLE1BQU0sSUFBSSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzFDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDRDQUE0QyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQzFELE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEQsTUFBTSxHQUFHLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQztRQUVsRCxNQUFNLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2xDLGlCQUFpQixDQUFDLE1BQU0sR0FBRyxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ3pDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGlDQUFpQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQy9DLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEQsTUFBTSxHQUFHLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRTVCLE1BQU0sQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMvQixpQkFBaUIsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzlDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDJDQUEyQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3pELE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEQsTUFBTSxHQUFHLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsY0FBYyxDQUFDLENBQUM7UUFFakQsTUFBTSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNsQyxpQkFBaUIsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzlDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGlDQUFpQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQy9DLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEQsTUFBTSxHQUFHLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRTVCLE1BQU0sQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMvQixpQkFBaUIsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNqRCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQywrQkFBK0IsRUFBRSxLQUFLLElBQUksRUFBRTtRQUM3QyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2xELE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBRTFCLE1BQU0sQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMvQixpQkFBaUIsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzlDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGtDQUFrQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ2hELE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEQsTUFBTSxHQUFHLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUUzQixNQUFNLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDL0IsaUJBQWlCLENBQUMsTUFBTSxHQUFHLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM5QyxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxtQ0FBbUMsRUFBRSxLQUFLLElBQUksRUFBRTtRQUNqRCxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2xELE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFL0IsTUFBTSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxDQUFDLENBQUM7UUFDOUIsaUJBQWlCLENBQUMsTUFBTSxHQUFHLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzNDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDhCQUE4QixFQUFFLEtBQUssSUFBSSxFQUFFO1FBQzVDLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEQsTUFBTSxHQUFHLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFakMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxDQUFDLENBQUM7UUFDOUIsaUJBQWlCLENBQUMsTUFBTSxHQUFHLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzNDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGlDQUFpQyxFQUFFLEdBQUcsRUFBRTtRQUN6QyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLElBQUksQ0FBQyxFQUFlLENBQUMsQ0FBQzthQUNqQyxZQUFZLENBQUMsZ0RBQWdELENBQUMsQ0FBQztJQUN0RSxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyw4QkFBOEIsRUFBRSxLQUFLLElBQUksRUFBRTtRQUM1QyxNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2pELGlCQUFpQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzVDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGdDQUFnQyxFQUFFLEdBQUcsRUFBRTtRQUN4QyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7YUFDdkIsWUFBWSxDQUFDLHNEQUFzRCxDQUFDLENBQUM7SUFDNUUsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0ICogYXMgdGYgZnJvbSAnLi4vaW5kZXgnO1xuaW1wb3J0IHtBTExfRU5WUywgZGVzY3JpYmVXaXRoRmxhZ3N9IGZyb20gJy4uL2phc21pbmVfdXRpbCc7XG5pbXBvcnQge2V4cGVjdEFycmF5c0Nsb3NlLCBleHBlY3RBcnJheXNFcXVhbH0gZnJvbSAnLi4vdGVzdF91dGlsJztcblxuZGVzY3JpYmVXaXRoRmxhZ3MoJ3Byb2QnLCBBTExfRU5WUywgKCkgPT4ge1xuICBpdCgnYmFzaWMnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgYSA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCAwLCAwLCAxXSwgWzMsIDJdKTtcbiAgICBjb25zdCByZXN1bHQgPSB0Zi5wcm9kKGEpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDApO1xuICB9KTtcblxuICBpdCgncHJvcGFnYXRlcyBOYU5zJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSB0Zi50ZW5zb3IyZChbMSwgMiwgMywgTmFOLCAwLCAxXSwgWzMsIDJdKTtcbiAgICBleHBlY3RBcnJheXNFcXVhbChhd2FpdCB0Zi5wcm9kKGEpLmRhdGEoKSwgTmFOKTtcbiAgfSk7XG5cbiAgaXQoJ3Byb2Qgb3ZlciBkdHlwZSBpbnQzMicsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBhID0gdGYudGVuc29yMWQoWzEsIDUsIDcsIDNdLCAnaW50MzInKTtcbiAgICBjb25zdCBwcm9kID0gdGYucHJvZChhKTtcbiAgICBleHBlY3RBcnJheXNFcXVhbChhd2FpdCBwcm9kLmRhdGEoKSwgMTA1KTtcbiAgfSk7XG5cbiAgaXQoJ3Byb2Qgb3ZlciBkdHlwZSBib29sJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbdHJ1ZSwgZmFsc2UsIGZhbHNlLCB0cnVlLCB0cnVlXSwgJ2Jvb2wnKTtcbiAgICBjb25zdCBwcm9kID0gdGYucHJvZChhKTtcbiAgICBleHBlY3RBcnJheXNFcXVhbChhd2FpdCBwcm9kLmRhdGEoKSwgMCk7XG4gIH0pO1xuXG4gIGl0KCdwcm9kcyBhbGwgdmFsdWVzIGluIDJEIGFycmF5IHdpdGgga2VlcCBkaW0nLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgYSA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCAxLCAwLCAxXSwgWzMsIDJdKTtcbiAgICBjb25zdCByZXMgPSB0Zi5wcm9kKGEsIG51bGwsIHRydWUgLyoga2VlcERpbXMgKi8pO1xuXG4gICAgZXhwZWN0KHJlcy5zaGFwZSkudG9FcXVhbChbMSwgMV0pO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlcy5kYXRhKCksIDApO1xuICB9KTtcblxuICBpdCgncHJvZHMgYWNyb3NzIGF4aXM9MCBpbiAyRCBhcnJheScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBhID0gdGYudGVuc29yMmQoWzEsIDIsIDMsIDEsIDAsIDFdLCBbMywgMl0pO1xuICAgIGNvbnN0IHJlcyA9IHRmLnByb2QoYSwgWzBdKTtcblxuICAgIGV4cGVjdChyZXMuc2hhcGUpLnRvRXF1YWwoWzJdKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXMuZGF0YSgpLCBbMCwgMl0pO1xuICB9KTtcblxuICBpdCgncHJvZHMgYWNyb3NzIGF4aXM9MCBpbiAyRCBhcnJheSwga2VlcERpbXMnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgYSA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCAxLCAwLCAxXSwgWzMsIDJdKTtcbiAgICBjb25zdCByZXMgPSB0Zi5wcm9kKGEsIFswXSwgdHJ1ZSAvKiBrZWVwRGltcyAqLyk7XG5cbiAgICBleHBlY3QocmVzLnNoYXBlKS50b0VxdWFsKFsxLCAyXSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzLmRhdGEoKSwgWzAsIDJdKTtcbiAgfSk7XG5cbiAgaXQoJ3Byb2RzIGFjcm9zcyBheGlzPTEgaW4gMkQgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgYSA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCAxLCAxLCAxXSwgWzMsIDJdKTtcbiAgICBjb25zdCByZXMgPSB0Zi5wcm9kKGEsIFsxXSk7XG5cbiAgICBleHBlY3QocmVzLnNoYXBlKS50b0VxdWFsKFszXSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzLmRhdGEoKSwgWzIsIDMsIDFdKTtcbiAgfSk7XG5cbiAgaXQoJzJELCBheGlzPTEgcHJvdmlkZWQgYXMgbnVtYmVyJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSB0Zi50ZW5zb3IyZChbMSwgMiwgMywgMSwgMSwgMV0sIFsyLCAzXSk7XG4gICAgY29uc3QgcmVzID0gdGYucHJvZChhLCAxKTtcblxuICAgIGV4cGVjdChyZXMuc2hhcGUpLnRvRXF1YWwoWzJdKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXMuZGF0YSgpLCBbNiwgMV0pO1xuICB9KTtcblxuICBpdCgnMkQsIGF4aXMgPSAtMSBwcm92aWRlZCBhcyBudW1iZXInLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgYSA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCAxLCAxLCAxXSwgWzIsIDNdKTtcbiAgICBjb25zdCByZXMgPSB0Zi5wcm9kKGEsIC0xKTtcblxuICAgIGV4cGVjdChyZXMuc2hhcGUpLnRvRXF1YWwoWzJdKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXMuZGF0YSgpLCBbNiwgMV0pO1xuICB9KTtcblxuICBpdCgncHJvZHMgYWNyb3NzIGF4aXM9MCwxIGluIDJEIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSB0Zi50ZW5zb3IyZChbMSwgMiwgMywgMSwgMSwgMV0sIFszLCAyXSk7XG4gICAgY29uc3QgcmVzID0gdGYucHJvZChhLCBbMCwgMV0pO1xuXG4gICAgZXhwZWN0KHJlcy5zaGFwZSkudG9FcXVhbChbXSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzLmRhdGEoKSwgWzZdKTtcbiAgfSk7XG5cbiAgaXQoJzJELCBheGlzPVstMSwtMl0gaW4gMkQgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgYSA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCAxLCAxLCAxXSwgWzMsIDJdKTtcbiAgICBjb25zdCByZXMgPSB0Zi5wcm9kKGEsIFstMSwgLTJdKTtcblxuICAgIGV4cGVjdChyZXMuc2hhcGUpLnRvRXF1YWwoW10pO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlcy5kYXRhKCksIFs2XSk7XG4gIH0pO1xuXG4gIGl0KCd0aHJvd3Mgd2hlbiBwYXNzZWQgYSBub24tdGVuc29yJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiB0Zi5wcm9kKHt9IGFzIHRmLlRlbnNvcikpXG4gICAgICAgIC50b1Rocm93RXJyb3IoL0FyZ3VtZW50ICd4JyBwYXNzZWQgdG8gJ3Byb2QnIG11c3QgYmUgYSBUZW5zb3IvKTtcbiAgfSk7XG5cbiAgaXQoJ2FjY2VwdHMgYSB0ZW5zb3ItbGlrZSBvYmplY3QnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgcmVzdWx0ID0gdGYucHJvZChbWzEsIDJdLCBbMywgMV0sIFsxLCAxXV0pO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDYpO1xuICB9KTtcblxuICBpdCgndGhyb3dzIGVycm9yIGZvciBzdHJpbmcgdGVuc29yJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiB0Zi5wcm9kKFsnYSddKSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcigvQXJndW1lbnQgJ3gnIHBhc3NlZCB0byAncHJvZCcgbXVzdCBiZSBudW1lcmljIHRlbnNvci8pO1xuICB9KTtcbn0pO1xuIl19