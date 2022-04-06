/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
import { util } from '..';
import * as tf from '../index';
import { ALL_ENVS, describeWithFlags } from '../jasmine_util';
import { expectValuesInRange } from '../test_util';
import { MPRandGauss, RandGamma, UniformRandom } from './rand_util';
import { expectArrayInMeanStdRange, jarqueBeraNormalityTest } from './rand_util';
describeWithFlags('rand', ALL_ENVS, () => {
    it('should return a random 1D float32 array', async () => {
        const shape = [10];
        // Enusre defaults to float32 w/o type:
        let result = tf.rand(shape, () => util.randUniform(0, 2));
        expect(result.dtype).toBe('float32');
        expectValuesInRange(await result.data(), 0, 2);
        result = tf.rand(shape, () => util.randUniform(0, 1.5));
        expect(result.dtype).toBe('float32');
        expectValuesInRange(await result.data(), 0, 1.5);
    });
    it('should return a random 1D int32 array', async () => {
        const shape = [10];
        const result = tf.rand(shape, () => util.randUniform(0, 2), 'int32');
        expect(result.dtype).toBe('int32');
        expectValuesInRange(await result.data(), 0, 2);
    });
    it('should return a random 1D bool array', async () => {
        const shape = [10];
        const result = tf.rand(shape, () => util.randUniform(0, 1), 'bool');
        expect(result.dtype).toBe('bool');
        expectValuesInRange(await result.data(), 0, 1);
    });
    it('should return a random 2D float32 array', async () => {
        const shape = [3, 4];
        // Enusre defaults to float32 w/o type:
        let result = tf.rand(shape, () => util.randUniform(0, 2.5));
        expect(result.dtype).toBe('float32');
        expectValuesInRange(await result.data(), 0, 2.5);
        result = tf.rand(shape, () => util.randUniform(0, 1.5), 'float32');
        expect(result.dtype).toBe('float32');
        expectValuesInRange(await result.data(), 0, 1.5);
    });
    it('should return a random 2D int32 array', async () => {
        const shape = [3, 4];
        const result = tf.rand(shape, () => util.randUniform(0, 2), 'int32');
        expect(result.dtype).toBe('int32');
        expectValuesInRange(await result.data(), 0, 2);
    });
    it('should return a random 2D bool array', async () => {
        const shape = [3, 4];
        const result = tf.rand(shape, () => util.randUniform(0, 1), 'bool');
        expect(result.dtype).toBe('bool');
        expectValuesInRange(await result.data(), 0, 1);
    });
    it('should return a random 3D float32 array', async () => {
        const shape = [3, 4, 5];
        // Enusre defaults to float32 w/o type:
        let result = tf.rand(shape, () => util.randUniform(0, 2.5));
        expect(result.dtype).toBe('float32');
        expectValuesInRange(await result.data(), 0, 2.5);
        result = tf.rand(shape, () => util.randUniform(0, 1.5), 'float32');
        expect(result.dtype).toBe('float32');
        expectValuesInRange(await result.data(), 0, 1.5);
    });
    it('should return a random 3D int32 array', async () => {
        const shape = [3, 4, 5];
        const result = tf.rand(shape, () => util.randUniform(0, 2), 'int32');
        expect(result.dtype).toBe('int32');
        expectValuesInRange(await result.data(), 0, 2);
    });
    it('should return a random 3D bool array', async () => {
        const shape = [3, 4, 5];
        const result = tf.rand(shape, () => util.randUniform(0, 1), 'bool');
        expect(result.dtype).toBe('bool');
        expectValuesInRange(await result.data(), 0, 1);
    });
    it('should return a random 4D float32 array', async () => {
        const shape = [3, 4, 5, 6];
        // Enusre defaults to float32 w/o type:
        let result = tf.rand(shape, () => util.randUniform(0, 2.5));
        expect(result.dtype).toBe('float32');
        expectValuesInRange(await result.data(), 0, 2.5);
        result = tf.rand(shape, () => util.randUniform(0, 1.5));
        expect(result.dtype).toBe('float32');
        expectValuesInRange(await result.data(), 0, 1.5);
    });
    it('should return a random 4D int32 array', async () => {
        const shape = [3, 4, 5, 6];
        const result = tf.rand(shape, () => util.randUniform(0, 2), 'int32');
        expect(result.dtype).toBe('int32');
        expectValuesInRange(await result.data(), 0, 2);
    });
    it('should return a random 4D bool array', async () => {
        const shape = [3, 4, 5, 6];
        const result = tf.rand(shape, () => util.randUniform(0, 1), 'bool');
        expect(result.dtype).toBe('bool');
        expectValuesInRange(await result.data(), 0, 1);
    });
});
function isFloat(n) {
    return Number(n) === n && n % 1 !== 0;
}
describe('MPRandGauss', () => {
    const EPSILON = 0.05;
    const SEED = 2002;
    it('should default to float32 numbers', () => {
        const rand = new MPRandGauss(0, 1.5);
        expect(isFloat(rand.nextValue())).toBe(true);
    });
    it('should handle a mean/stdv of float32 numbers', () => {
        const rand = new MPRandGauss(0, 1.5, 'float32', false /* truncated */, SEED);
        const values = [];
        const size = 10000;
        for (let i = 0; i < size; i++) {
            values.push(rand.nextValue());
        }
        expectArrayInMeanStdRange(values, 0, 1.5, EPSILON);
        jarqueBeraNormalityTest(values);
    });
    it('should handle int32 numbers', () => {
        const rand = new MPRandGauss(0, 1, 'int32');
        expect(isFloat(rand.nextValue())).toBe(false);
    });
    it('should handle a mean/stdv of int32 numbers', () => {
        const rand = new MPRandGauss(0, 2, 'int32', false /* truncated */, SEED);
        const values = [];
        const size = 10000;
        for (let i = 0; i < size; i++) {
            values.push(rand.nextValue());
        }
        expectArrayInMeanStdRange(values, 0, 2, EPSILON);
        jarqueBeraNormalityTest(values);
    });
    it('Should not have a more than 2x std-d from mean for truncated values', () => {
        const stdv = 1.5;
        const rand = new MPRandGauss(0, stdv, 'float32', true /* truncated */);
        for (let i = 0; i < 1000; i++) {
            expect(Math.abs(rand.nextValue())).toBeLessThan(stdv * 2);
        }
    });
});
describe('RandGamma', () => {
    const SEED = 2002;
    it('should default to float32 numbers', () => {
        const rand = new RandGamma(2, 2, 'float32');
        expect(isFloat(rand.nextValue())).toBe(true);
    });
    it('should handle an alpha/beta of float32 numbers', () => {
        const rand = new RandGamma(2, 2, 'float32', SEED);
        const values = [];
        const size = 10000;
        for (let i = 0; i < size; i++) {
            values.push(rand.nextValue());
        }
        expectValuesInRange(values, 0, 30);
    });
    it('should handle int32 numbers', () => {
        const rand = new RandGamma(2, 2, 'int32');
        expect(isFloat(rand.nextValue())).toBe(false);
    });
    it('should handle an alpha/beta of int32 numbers', () => {
        const rand = new RandGamma(2, 2, 'int32', SEED);
        const values = [];
        const size = 10000;
        for (let i = 0; i < size; i++) {
            values.push(rand.nextValue());
        }
        expectValuesInRange(values, 0, 30);
    });
});
describe('UniformRandom', () => {
    it('float32, no seed', () => {
        const min = 0.2;
        const max = 0.24;
        const dtype = 'float32';
        const xs = [];
        for (let i = 0; i < 10; ++i) {
            const rand = new UniformRandom(min, max, dtype);
            const x = rand.nextValue();
            xs.push(x);
        }
        expect(Math.min(...xs)).toBeGreaterThanOrEqual(min);
        expect(Math.max(...xs)).toBeLessThan(max);
    });
    it('int32, no seed', () => {
        const min = 13;
        const max = 37;
        const dtype = 'int32';
        const xs = [];
        for (let i = 0; i < 10; ++i) {
            const rand = new UniformRandom(min, max, dtype);
            const x = rand.nextValue();
            expect(Number.isInteger(x)).toEqual(true);
            xs.push(x);
        }
        expect(Math.min(...xs)).toBeGreaterThanOrEqual(min);
        expect(Math.max(...xs)).toBeLessThanOrEqual(max);
    });
    it('seed is number', () => {
        const min = -1.2;
        const max = -0.4;
        const dtype = 'float32';
        const seed = 1337;
        const xs = [];
        for (let i = 0; i < 10; ++i) {
            const rand = new UniformRandom(min, max, dtype, seed);
            const x = rand.nextValue();
            expect(x).toBeGreaterThanOrEqual(min);
            expect(x).toBeLessThan(max);
            xs.push(x);
        }
        // Assert deterministic results.
        expect(Math.min(...xs)).toEqual(Math.max(...xs));
    });
    it('seed === null', () => {
        const min = 0;
        const max = 1;
        const dtype = 'float32';
        const seed = null;
        const rand = new UniformRandom(min, max, dtype, seed);
        const x = rand.nextValue();
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThan(1);
    });
    it('seed === undefined', () => {
        const min = 0;
        const max = 1;
        const dtype = 'float32';
        const seed = undefined;
        const rand = new UniformRandom(min, max, dtype, seed);
        const x = rand.nextValue();
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThan(1);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmFuZF90ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvcmFuZF90ZXN0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxJQUFJLENBQUM7QUFDeEIsT0FBTyxLQUFLLEVBQUUsTUFBTSxVQUFVLENBQUM7QUFDL0IsT0FBTyxFQUFDLFFBQVEsRUFBRSxpQkFBaUIsRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBQzVELE9BQU8sRUFBQyxtQkFBbUIsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUVqRCxPQUFPLEVBQUMsV0FBVyxFQUFFLFNBQVMsRUFBRSxhQUFhLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDbEUsT0FBTyxFQUFDLHlCQUF5QixFQUFFLHVCQUF1QixFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9FLGlCQUFpQixDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsR0FBRyxFQUFFO0lBQ3ZDLEVBQUUsQ0FBQyx5Q0FBeUMsRUFBRSxLQUFLLElBQUksRUFBRTtRQUN2RCxNQUFNLEtBQUssR0FBYSxDQUFDLEVBQUUsQ0FBQyxDQUFDO1FBRTdCLHVDQUF1QztRQUN2QyxJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLEtBQUssRUFBRSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzFELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQ3JDLG1CQUFtQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUUvQyxNQUFNLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxLQUFLLEVBQUUsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQztRQUN4RCxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUNyQyxtQkFBbUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUM7SUFDbkQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsdUNBQXVDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDckQsTUFBTSxLQUFLLEdBQWEsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUM3QixNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLEtBQUssRUFBRSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztRQUNyRSxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUNuQyxtQkFBbUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDakQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsc0NBQXNDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDcEQsTUFBTSxLQUFLLEdBQWEsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUM3QixNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLEtBQUssRUFBRSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxNQUFNLENBQUMsQ0FBQztRQUNwRSxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUNsQyxtQkFBbUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDakQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMseUNBQXlDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDdkQsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFFckIsdUNBQXVDO1FBQ3ZDLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsS0FBSyxFQUFFLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUM7UUFDNUQsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDckMsbUJBQW1CLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBRWpELE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLEtBQUssRUFBRSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsRUFBRSxTQUFTLENBQUMsQ0FBQztRQUNuRSxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUNyQyxtQkFBbUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUM7SUFDbkQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsdUNBQXVDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDckQsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckIsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxLQUFLLEVBQUUsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFDckUsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUM7UUFDbkMsbUJBQW1CLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ2pELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHNDQUFzQyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3BELE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JCLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsS0FBSyxFQUFFLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDO1FBQ3BFLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ2xDLG1CQUFtQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNqRCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyx5Q0FBeUMsRUFBRSxLQUFLLElBQUksRUFBRTtRQUN2RCxNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFFeEIsdUNBQXVDO1FBQ3ZDLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsS0FBSyxFQUFFLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUM7UUFDNUQsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDckMsbUJBQW1CLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBRWpELE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLEtBQUssRUFBRSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsRUFBRSxTQUFTLENBQUMsQ0FBQztRQUNuRSxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUNyQyxtQkFBbUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUM7SUFDbkQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsdUNBQXVDLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDckQsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsS0FBSyxFQUFFLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQ3JFLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQ25DLG1CQUFtQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNqRCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxzQ0FBc0MsRUFBRSxLQUFLLElBQUksRUFBRTtRQUNwRCxNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDeEIsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxLQUFLLEVBQUUsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUM7UUFDcEUsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDbEMsbUJBQW1CLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ2pELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHlDQUF5QyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3ZELE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFFM0IsdUNBQXVDO1FBQ3ZDLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsS0FBSyxFQUFFLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUM7UUFDNUQsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDckMsbUJBQW1CLENBQUMsTUFBTSxNQUFNLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBRWpELE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLEtBQUssRUFBRSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQyxDQUFDO1FBQ3hELE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQ3JDLG1CQUFtQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQztJQUNuRCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyx1Q0FBdUMsRUFBRSxLQUFLLElBQUksRUFBRTtRQUNyRCxNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQzNCLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsS0FBSyxFQUFFLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQ3JFLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQ25DLG1CQUFtQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNqRCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxzQ0FBc0MsRUFBRSxLQUFLLElBQUksRUFBRTtRQUNwRCxNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQzNCLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsS0FBSyxFQUFFLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDO1FBQ3BFLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ2xDLG1CQUFtQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNqRCxDQUFDLENBQUMsQ0FBQztBQUNMLENBQUMsQ0FBQyxDQUFDO0FBRUgsU0FBUyxPQUFPLENBQUMsQ0FBUztJQUN4QixPQUFPLE1BQU0sQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELFFBQVEsQ0FBQyxhQUFhLEVBQUUsR0FBRyxFQUFFO0lBQzNCLE1BQU0sT0FBTyxHQUFHLElBQUksQ0FBQztJQUNyQixNQUFNLElBQUksR0FBRyxJQUFJLENBQUM7SUFFbEIsRUFBRSxDQUFDLG1DQUFtQyxFQUFFLEdBQUcsRUFBRTtRQUMzQyxNQUFNLElBQUksR0FBRyxJQUFJLFdBQVcsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDckMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsU0FBUyxFQUFFLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUMvQyxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyw4Q0FBOEMsRUFBRSxHQUFHLEVBQUU7UUFDdEQsTUFBTSxJQUFJLEdBQ04sSUFBSSxXQUFXLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxTQUFTLEVBQUUsS0FBSyxDQUFDLGVBQWUsRUFBRSxJQUFJLENBQUMsQ0FBQztRQUNwRSxNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUM7UUFDbEIsTUFBTSxJQUFJLEdBQUcsS0FBSyxDQUFDO1FBQ25CLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLEVBQUUsQ0FBQyxFQUFFLEVBQUU7WUFDN0IsTUFBTSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsU0FBUyxFQUFFLENBQUMsQ0FBQztTQUMvQjtRQUNELHlCQUF5QixDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsR0FBRyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQ25ELHVCQUF1QixDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBQ2xDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDZCQUE2QixFQUFFLEdBQUcsRUFBRTtRQUNyQyxNQUFNLElBQUksR0FBRyxJQUFJLFdBQVcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQzVDLE1BQU0sQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLFNBQVMsRUFBRSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDaEQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsNENBQTRDLEVBQUUsR0FBRyxFQUFFO1FBQ3BELE1BQU0sSUFBSSxHQUFHLElBQUksV0FBVyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsT0FBTyxFQUFFLEtBQUssQ0FBQyxlQUFlLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDekUsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDO1FBQ2xCLE1BQU0sSUFBSSxHQUFHLEtBQUssQ0FBQztRQUNuQixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxFQUFFLENBQUMsRUFBRSxFQUFFO1lBQzdCLE1BQU0sQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFNBQVMsRUFBRSxDQUFDLENBQUM7U0FDL0I7UUFDRCx5QkFBeUIsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztRQUNqRCx1QkFBdUIsQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUNsQyxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxxRUFBcUUsRUFDckUsR0FBRyxFQUFFO1FBQ0gsTUFBTSxJQUFJLEdBQUcsR0FBRyxDQUFDO1FBQ2pCLE1BQU0sSUFBSSxHQUFHLElBQUksV0FBVyxDQUFDLENBQUMsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLElBQUksQ0FBQyxlQUFlLENBQUMsQ0FBQztRQUN2RSxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxFQUFFLENBQUMsRUFBRSxFQUFFO1lBQzdCLE1BQU0sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxTQUFTLEVBQUUsQ0FBQyxDQUFDLENBQUMsWUFBWSxDQUFDLElBQUksR0FBRyxDQUFDLENBQUMsQ0FBQztTQUMzRDtJQUNILENBQUMsQ0FBQyxDQUFDO0FBQ1IsQ0FBQyxDQUFDLENBQUM7QUFFSCxRQUFRLENBQUMsV0FBVyxFQUFFLEdBQUcsRUFBRTtJQUN6QixNQUFNLElBQUksR0FBRyxJQUFJLENBQUM7SUFFbEIsRUFBRSxDQUFDLG1DQUFtQyxFQUFFLEdBQUcsRUFBRTtRQUMzQyxNQUFNLElBQUksR0FBRyxJQUFJLFNBQVMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLFNBQVMsQ0FBQyxDQUFDO1FBQzVDLE1BQU0sQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLFNBQVMsRUFBRSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDL0MsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsZ0RBQWdELEVBQUUsR0FBRyxFQUFFO1FBQ3hELE1BQU0sSUFBSSxHQUFHLElBQUksU0FBUyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsU0FBUyxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQ2xELE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQztRQUNsQixNQUFNLElBQUksR0FBRyxLQUFLLENBQUM7UUFDbkIsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksRUFBRSxDQUFDLEVBQUUsRUFBRTtZQUM3QixNQUFNLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxTQUFTLEVBQUUsQ0FBQyxDQUFDO1NBQy9CO1FBQ0QsbUJBQW1CLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQztJQUNyQyxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyw2QkFBNkIsRUFBRSxHQUFHLEVBQUU7UUFDckMsTUFBTSxJQUFJLEdBQUcsSUFBSSxTQUFTLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztRQUMxQyxNQUFNLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQyxTQUFTLEVBQUUsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQ2hELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDhDQUE4QyxFQUFFLEdBQUcsRUFBRTtRQUN0RCxNQUFNLElBQUksR0FBRyxJQUFJLFNBQVMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLE9BQU8sRUFBRSxJQUFJLENBQUMsQ0FBQztRQUNoRCxNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUM7UUFDbEIsTUFBTSxJQUFJLEdBQUcsS0FBSyxDQUFDO1FBQ25CLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLEVBQUUsQ0FBQyxFQUFFLEVBQUU7WUFDN0IsTUFBTSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsU0FBUyxFQUFFLENBQUMsQ0FBQztTQUMvQjtRQUNELG1CQUFtQixDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUM7SUFDckMsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQztBQUVILFFBQVEsQ0FBQyxlQUFlLEVBQUUsR0FBRyxFQUFFO0lBQzdCLEVBQUUsQ0FBQyxrQkFBa0IsRUFBRSxHQUFHLEVBQUU7UUFDMUIsTUFBTSxHQUFHLEdBQUcsR0FBRyxDQUFDO1FBQ2hCLE1BQU0sR0FBRyxHQUFHLElBQUksQ0FBQztRQUNqQixNQUFNLEtBQUssR0FBRyxTQUFTLENBQUM7UUFDeEIsTUFBTSxFQUFFLEdBQWEsRUFBRSxDQUFDO1FBQ3hCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUU7WUFDM0IsTUFBTSxJQUFJLEdBQUcsSUFBSSxhQUFhLENBQUMsR0FBRyxFQUFFLEdBQUcsRUFBRSxLQUFLLENBQUMsQ0FBQztZQUNoRCxNQUFNLENBQUMsR0FBRyxJQUFJLENBQUMsU0FBUyxFQUFFLENBQUM7WUFDM0IsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUNaO1FBQ0QsTUFBTSxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLHNCQUFzQixDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBQ3BELE1BQU0sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxZQUFZLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDNUMsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsZ0JBQWdCLEVBQUUsR0FBRyxFQUFFO1FBQ3hCLE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQztRQUNmLE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQztRQUNmLE1BQU0sS0FBSyxHQUFHLE9BQU8sQ0FBQztRQUN0QixNQUFNLEVBQUUsR0FBYSxFQUFFLENBQUM7UUFDeEIsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRTtZQUMzQixNQUFNLElBQUksR0FBRyxJQUFJLGFBQWEsQ0FBQyxHQUFHLEVBQUUsR0FBRyxFQUFFLEtBQUssQ0FBQyxDQUFDO1lBQ2hELE1BQU0sQ0FBQyxHQUFHLElBQUksQ0FBQyxTQUFTLEVBQUUsQ0FBQztZQUMzQixNQUFNLENBQUMsTUFBTSxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsQ0FBQztZQUMxQyxFQUFFLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBQ1o7UUFDRCxNQUFNLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsc0JBQXNCLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDcEQsTUFBTSxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLG1CQUFtQixDQUFDLEdBQUcsQ0FBQyxDQUFDO0lBQ25ELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGdCQUFnQixFQUFFLEdBQUcsRUFBRTtRQUN4QixNQUFNLEdBQUcsR0FBRyxDQUFDLEdBQUcsQ0FBQztRQUNqQixNQUFNLEdBQUcsR0FBRyxDQUFDLEdBQUcsQ0FBQztRQUNqQixNQUFNLEtBQUssR0FBRyxTQUFTLENBQUM7UUFDeEIsTUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDO1FBQ2xCLE1BQU0sRUFBRSxHQUFhLEVBQUUsQ0FBQztRQUN4QixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFO1lBQzNCLE1BQU0sSUFBSSxHQUFHLElBQUksYUFBYSxDQUFDLEdBQUcsRUFBRSxHQUFHLEVBQUUsS0FBSyxFQUFFLElBQUksQ0FBQyxDQUFDO1lBQ3RELE1BQU0sQ0FBQyxHQUFHLElBQUksQ0FBQyxTQUFTLEVBQUUsQ0FBQztZQUMzQixNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsc0JBQXNCLENBQUMsR0FBRyxDQUFDLENBQUM7WUFDdEMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLFlBQVksQ0FBQyxHQUFHLENBQUMsQ0FBQztZQUM1QixFQUFFLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBQ1o7UUFDRCxnQ0FBZ0M7UUFDaEMsTUFBTSxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNuRCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxlQUFlLEVBQUUsR0FBRyxFQUFFO1FBQ3ZCLE1BQU0sR0FBRyxHQUFHLENBQUMsQ0FBQztRQUNkLE1BQU0sR0FBRyxHQUFHLENBQUMsQ0FBQztRQUNkLE1BQU0sS0FBSyxHQUFHLFNBQVMsQ0FBQztRQUN4QixNQUFNLElBQUksR0FBVyxJQUFJLENBQUM7UUFDMUIsTUFBTSxJQUFJLEdBQUcsSUFBSSxhQUFhLENBQUMsR0FBRyxFQUFFLEdBQUcsRUFBRSxLQUFLLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDdEQsTUFBTSxDQUFDLEdBQUcsSUFBSSxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQzNCLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxzQkFBc0IsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNwQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzVCLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLG9CQUFvQixFQUFFLEdBQUcsRUFBRTtRQUM1QixNQUFNLEdBQUcsR0FBRyxDQUFDLENBQUM7UUFDZCxNQUFNLEdBQUcsR0FBRyxDQUFDLENBQUM7UUFDZCxNQUFNLEtBQUssR0FBRyxTQUFTLENBQUM7UUFDeEIsTUFBTSxJQUFJLEdBQVcsU0FBUyxDQUFDO1FBQy9CLE1BQU0sSUFBSSxHQUFHLElBQUksYUFBYSxDQUFDLEdBQUcsRUFBRSxHQUFHLEVBQUUsS0FBSyxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQ3RELE1BQU0sQ0FBQyxHQUFHLElBQUksQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUMzQixNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsc0JBQXNCLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDcEMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM1QixDQUFDLENBQUMsQ0FBQztBQUNMLENBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge3V0aWx9IGZyb20gJy4uJztcbmltcG9ydCAqIGFzIHRmIGZyb20gJy4uL2luZGV4JztcbmltcG9ydCB7QUxMX0VOVlMsIGRlc2NyaWJlV2l0aEZsYWdzfSBmcm9tICcuLi9qYXNtaW5lX3V0aWwnO1xuaW1wb3J0IHtleHBlY3RWYWx1ZXNJblJhbmdlfSBmcm9tICcuLi90ZXN0X3V0aWwnO1xuXG5pbXBvcnQge01QUmFuZEdhdXNzLCBSYW5kR2FtbWEsIFVuaWZvcm1SYW5kb219IGZyb20gJy4vcmFuZF91dGlsJztcbmltcG9ydCB7ZXhwZWN0QXJyYXlJbk1lYW5TdGRSYW5nZSwgamFycXVlQmVyYU5vcm1hbGl0eVRlc3R9IGZyb20gJy4vcmFuZF91dGlsJztcblxuZGVzY3JpYmVXaXRoRmxhZ3MoJ3JhbmQnLCBBTExfRU5WUywgKCkgPT4ge1xuICBpdCgnc2hvdWxkIHJldHVybiBhIHJhbmRvbSAxRCBmbG9hdDMyIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHNoYXBlOiBbbnVtYmVyXSA9IFsxMF07XG5cbiAgICAvLyBFbnVzcmUgZGVmYXVsdHMgdG8gZmxvYXQzMiB3L28gdHlwZTpcbiAgICBsZXQgcmVzdWx0ID0gdGYucmFuZChzaGFwZSwgKCkgPT4gdXRpbC5yYW5kVW5pZm9ybSgwLCAyKSk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnZmxvYXQzMicpO1xuICAgIGV4cGVjdFZhbHVlc0luUmFuZ2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgMik7XG5cbiAgICByZXN1bHQgPSB0Zi5yYW5kKHNoYXBlLCAoKSA9PiB1dGlsLnJhbmRVbmlmb3JtKDAsIDEuNSkpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvQmUoJ2Zsb2F0MzInKTtcbiAgICBleHBlY3RWYWx1ZXNJblJhbmdlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDEuNSk7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgcmV0dXJuIGEgcmFuZG9tIDFEIGludDMyIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHNoYXBlOiBbbnVtYmVyXSA9IFsxMF07XG4gICAgY29uc3QgcmVzdWx0ID0gdGYucmFuZChzaGFwZSwgKCkgPT4gdXRpbC5yYW5kVW5pZm9ybSgwLCAyKSwgJ2ludDMyJyk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnaW50MzInKTtcbiAgICBleHBlY3RWYWx1ZXNJblJhbmdlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDIpO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIHJldHVybiBhIHJhbmRvbSAxRCBib29sIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHNoYXBlOiBbbnVtYmVyXSA9IFsxMF07XG4gICAgY29uc3QgcmVzdWx0ID0gdGYucmFuZChzaGFwZSwgKCkgPT4gdXRpbC5yYW5kVW5pZm9ybSgwLCAxKSwgJ2Jvb2wnKTtcbiAgICBleHBlY3QocmVzdWx0LmR0eXBlKS50b0JlKCdib29sJyk7XG4gICAgZXhwZWN0VmFsdWVzSW5SYW5nZShhd2FpdCByZXN1bHQuZGF0YSgpLCAwLCAxKTtcbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCByZXR1cm4gYSByYW5kb20gMkQgZmxvYXQzMiBhcnJheScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBzaGFwZSA9IFszLCA0XTtcblxuICAgIC8vIEVudXNyZSBkZWZhdWx0cyB0byBmbG9hdDMyIHcvbyB0eXBlOlxuICAgIGxldCByZXN1bHQgPSB0Zi5yYW5kKHNoYXBlLCAoKSA9PiB1dGlsLnJhbmRVbmlmb3JtKDAsIDIuNSkpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvQmUoJ2Zsb2F0MzInKTtcbiAgICBleHBlY3RWYWx1ZXNJblJhbmdlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDIuNSk7XG5cbiAgICByZXN1bHQgPSB0Zi5yYW5kKHNoYXBlLCAoKSA9PiB1dGlsLnJhbmRVbmlmb3JtKDAsIDEuNSksICdmbG9hdDMyJyk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnZmxvYXQzMicpO1xuICAgIGV4cGVjdFZhbHVlc0luUmFuZ2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgMS41KTtcbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCByZXR1cm4gYSByYW5kb20gMkQgaW50MzIgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgc2hhcGUgPSBbMywgNF07XG4gICAgY29uc3QgcmVzdWx0ID0gdGYucmFuZChzaGFwZSwgKCkgPT4gdXRpbC5yYW5kVW5pZm9ybSgwLCAyKSwgJ2ludDMyJyk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnaW50MzInKTtcbiAgICBleHBlY3RWYWx1ZXNJblJhbmdlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDIpO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIHJldHVybiBhIHJhbmRvbSAyRCBib29sIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHNoYXBlID0gWzMsIDRdO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRmLnJhbmQoc2hhcGUsICgpID0+IHV0aWwucmFuZFVuaWZvcm0oMCwgMSksICdib29sJyk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnYm9vbCcpO1xuICAgIGV4cGVjdFZhbHVlc0luUmFuZ2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgMSk7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgcmV0dXJuIGEgcmFuZG9tIDNEIGZsb2F0MzIgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgc2hhcGUgPSBbMywgNCwgNV07XG5cbiAgICAvLyBFbnVzcmUgZGVmYXVsdHMgdG8gZmxvYXQzMiB3L28gdHlwZTpcbiAgICBsZXQgcmVzdWx0ID0gdGYucmFuZChzaGFwZSwgKCkgPT4gdXRpbC5yYW5kVW5pZm9ybSgwLCAyLjUpKTtcbiAgICBleHBlY3QocmVzdWx0LmR0eXBlKS50b0JlKCdmbG9hdDMyJyk7XG4gICAgZXhwZWN0VmFsdWVzSW5SYW5nZShhd2FpdCByZXN1bHQuZGF0YSgpLCAwLCAyLjUpO1xuXG4gICAgcmVzdWx0ID0gdGYucmFuZChzaGFwZSwgKCkgPT4gdXRpbC5yYW5kVW5pZm9ybSgwLCAxLjUpLCAnZmxvYXQzMicpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvQmUoJ2Zsb2F0MzInKTtcbiAgICBleHBlY3RWYWx1ZXNJblJhbmdlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDEuNSk7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgcmV0dXJuIGEgcmFuZG9tIDNEIGludDMyIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHNoYXBlID0gWzMsIDQsIDVdO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRmLnJhbmQoc2hhcGUsICgpID0+IHV0aWwucmFuZFVuaWZvcm0oMCwgMiksICdpbnQzMicpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvQmUoJ2ludDMyJyk7XG4gICAgZXhwZWN0VmFsdWVzSW5SYW5nZShhd2FpdCByZXN1bHQuZGF0YSgpLCAwLCAyKTtcbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCByZXR1cm4gYSByYW5kb20gM0QgYm9vbCBhcnJheScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBzaGFwZSA9IFszLCA0LCA1XTtcbiAgICBjb25zdCByZXN1bHQgPSB0Zi5yYW5kKHNoYXBlLCAoKSA9PiB1dGlsLnJhbmRVbmlmb3JtKDAsIDEpLCAnYm9vbCcpO1xuICAgIGV4cGVjdChyZXN1bHQuZHR5cGUpLnRvQmUoJ2Jvb2wnKTtcbiAgICBleHBlY3RWYWx1ZXNJblJhbmdlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDEpO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIHJldHVybiBhIHJhbmRvbSA0RCBmbG9hdDMyIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHNoYXBlID0gWzMsIDQsIDUsIDZdO1xuXG4gICAgLy8gRW51c3JlIGRlZmF1bHRzIHRvIGZsb2F0MzIgdy9vIHR5cGU6XG4gICAgbGV0IHJlc3VsdCA9IHRmLnJhbmQoc2hhcGUsICgpID0+IHV0aWwucmFuZFVuaWZvcm0oMCwgMi41KSk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnZmxvYXQzMicpO1xuICAgIGV4cGVjdFZhbHVlc0luUmFuZ2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgMi41KTtcblxuICAgIHJlc3VsdCA9IHRmLnJhbmQoc2hhcGUsICgpID0+IHV0aWwucmFuZFVuaWZvcm0oMCwgMS41KSk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnZmxvYXQzMicpO1xuICAgIGV4cGVjdFZhbHVlc0luUmFuZ2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgMS41KTtcbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCByZXR1cm4gYSByYW5kb20gNEQgaW50MzIgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgc2hhcGUgPSBbMywgNCwgNSwgNl07XG4gICAgY29uc3QgcmVzdWx0ID0gdGYucmFuZChzaGFwZSwgKCkgPT4gdXRpbC5yYW5kVW5pZm9ybSgwLCAyKSwgJ2ludDMyJyk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnaW50MzInKTtcbiAgICBleHBlY3RWYWx1ZXNJblJhbmdlKGF3YWl0IHJlc3VsdC5kYXRhKCksIDAsIDIpO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIHJldHVybiBhIHJhbmRvbSA0RCBib29sIGFycmF5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHNoYXBlID0gWzMsIDQsIDUsIDZdO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRmLnJhbmQoc2hhcGUsICgpID0+IHV0aWwucmFuZFVuaWZvcm0oMCwgMSksICdib29sJyk7XG4gICAgZXhwZWN0KHJlc3VsdC5kdHlwZSkudG9CZSgnYm9vbCcpO1xuICAgIGV4cGVjdFZhbHVlc0luUmFuZ2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgMCwgMSk7XG4gIH0pO1xufSk7XG5cbmZ1bmN0aW9uIGlzRmxvYXQobjogbnVtYmVyKTogYm9vbGVhbiB7XG4gIHJldHVybiBOdW1iZXIobikgPT09IG4gJiYgbiAlIDEgIT09IDA7XG59XG5cbmRlc2NyaWJlKCdNUFJhbmRHYXVzcycsICgpID0+IHtcbiAgY29uc3QgRVBTSUxPTiA9IDAuMDU7XG4gIGNvbnN0IFNFRUQgPSAyMDAyO1xuXG4gIGl0KCdzaG91bGQgZGVmYXVsdCB0byBmbG9hdDMyIG51bWJlcnMnLCAoKSA9PiB7XG4gICAgY29uc3QgcmFuZCA9IG5ldyBNUFJhbmRHYXVzcygwLCAxLjUpO1xuICAgIGV4cGVjdChpc0Zsb2F0KHJhbmQubmV4dFZhbHVlKCkpKS50b0JlKHRydWUpO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIGhhbmRsZSBhIG1lYW4vc3RkdiBvZiBmbG9hdDMyIG51bWJlcnMnLCAoKSA9PiB7XG4gICAgY29uc3QgcmFuZCA9XG4gICAgICAgIG5ldyBNUFJhbmRHYXVzcygwLCAxLjUsICdmbG9hdDMyJywgZmFsc2UgLyogdHJ1bmNhdGVkICovLCBTRUVEKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBbXTtcbiAgICBjb25zdCBzaXplID0gMTAwMDA7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBzaXplOyBpKyspIHtcbiAgICAgIHZhbHVlcy5wdXNoKHJhbmQubmV4dFZhbHVlKCkpO1xuICAgIH1cbiAgICBleHBlY3RBcnJheUluTWVhblN0ZFJhbmdlKHZhbHVlcywgMCwgMS41LCBFUFNJTE9OKTtcbiAgICBqYXJxdWVCZXJhTm9ybWFsaXR5VGVzdCh2YWx1ZXMpO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIGhhbmRsZSBpbnQzMiBudW1iZXJzJywgKCkgPT4ge1xuICAgIGNvbnN0IHJhbmQgPSBuZXcgTVBSYW5kR2F1c3MoMCwgMSwgJ2ludDMyJyk7XG4gICAgZXhwZWN0KGlzRmxvYXQocmFuZC5uZXh0VmFsdWUoKSkpLnRvQmUoZmFsc2UpO1xuICB9KTtcblxuICBpdCgnc2hvdWxkIGhhbmRsZSBhIG1lYW4vc3RkdiBvZiBpbnQzMiBudW1iZXJzJywgKCkgPT4ge1xuICAgIGNvbnN0IHJhbmQgPSBuZXcgTVBSYW5kR2F1c3MoMCwgMiwgJ2ludDMyJywgZmFsc2UgLyogdHJ1bmNhdGVkICovLCBTRUVEKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBbXTtcbiAgICBjb25zdCBzaXplID0gMTAwMDA7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBzaXplOyBpKyspIHtcbiAgICAgIHZhbHVlcy5wdXNoKHJhbmQubmV4dFZhbHVlKCkpO1xuICAgIH1cbiAgICBleHBlY3RBcnJheUluTWVhblN0ZFJhbmdlKHZhbHVlcywgMCwgMiwgRVBTSUxPTik7XG4gICAgamFycXVlQmVyYU5vcm1hbGl0eVRlc3QodmFsdWVzKTtcbiAgfSk7XG5cbiAgaXQoJ1Nob3VsZCBub3QgaGF2ZSBhIG1vcmUgdGhhbiAyeCBzdGQtZCBmcm9tIG1lYW4gZm9yIHRydW5jYXRlZCB2YWx1ZXMnLFxuICAgICAoKSA9PiB7XG4gICAgICAgY29uc3Qgc3RkdiA9IDEuNTtcbiAgICAgICBjb25zdCByYW5kID0gbmV3IE1QUmFuZEdhdXNzKDAsIHN0ZHYsICdmbG9hdDMyJywgdHJ1ZSAvKiB0cnVuY2F0ZWQgKi8pO1xuICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgMTAwMDsgaSsrKSB7XG4gICAgICAgICBleHBlY3QoTWF0aC5hYnMocmFuZC5uZXh0VmFsdWUoKSkpLnRvQmVMZXNzVGhhbihzdGR2ICogMik7XG4gICAgICAgfVxuICAgICB9KTtcbn0pO1xuXG5kZXNjcmliZSgnUmFuZEdhbW1hJywgKCkgPT4ge1xuICBjb25zdCBTRUVEID0gMjAwMjtcblxuICBpdCgnc2hvdWxkIGRlZmF1bHQgdG8gZmxvYXQzMiBudW1iZXJzJywgKCkgPT4ge1xuICAgIGNvbnN0IHJhbmQgPSBuZXcgUmFuZEdhbW1hKDIsIDIsICdmbG9hdDMyJyk7XG4gICAgZXhwZWN0KGlzRmxvYXQocmFuZC5uZXh0VmFsdWUoKSkpLnRvQmUodHJ1ZSk7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgaGFuZGxlIGFuIGFscGhhL2JldGEgb2YgZmxvYXQzMiBudW1iZXJzJywgKCkgPT4ge1xuICAgIGNvbnN0IHJhbmQgPSBuZXcgUmFuZEdhbW1hKDIsIDIsICdmbG9hdDMyJywgU0VFRCk7XG4gICAgY29uc3QgdmFsdWVzID0gW107XG4gICAgY29uc3Qgc2l6ZSA9IDEwMDAwO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgc2l6ZTsgaSsrKSB7XG4gICAgICB2YWx1ZXMucHVzaChyYW5kLm5leHRWYWx1ZSgpKTtcbiAgICB9XG4gICAgZXhwZWN0VmFsdWVzSW5SYW5nZSh2YWx1ZXMsIDAsIDMwKTtcbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCBoYW5kbGUgaW50MzIgbnVtYmVycycsICgpID0+IHtcbiAgICBjb25zdCByYW5kID0gbmV3IFJhbmRHYW1tYSgyLCAyLCAnaW50MzInKTtcbiAgICBleHBlY3QoaXNGbG9hdChyYW5kLm5leHRWYWx1ZSgpKSkudG9CZShmYWxzZSk7XG4gIH0pO1xuXG4gIGl0KCdzaG91bGQgaGFuZGxlIGFuIGFscGhhL2JldGEgb2YgaW50MzIgbnVtYmVycycsICgpID0+IHtcbiAgICBjb25zdCByYW5kID0gbmV3IFJhbmRHYW1tYSgyLCAyLCAnaW50MzInLCBTRUVEKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBbXTtcbiAgICBjb25zdCBzaXplID0gMTAwMDA7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBzaXplOyBpKyspIHtcbiAgICAgIHZhbHVlcy5wdXNoKHJhbmQubmV4dFZhbHVlKCkpO1xuICAgIH1cbiAgICBleHBlY3RWYWx1ZXNJblJhbmdlKHZhbHVlcywgMCwgMzApO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZSgnVW5pZm9ybVJhbmRvbScsICgpID0+IHtcbiAgaXQoJ2Zsb2F0MzIsIG5vIHNlZWQnLCAoKSA9PiB7XG4gICAgY29uc3QgbWluID0gMC4yO1xuICAgIGNvbnN0IG1heCA9IDAuMjQ7XG4gICAgY29uc3QgZHR5cGUgPSAnZmxvYXQzMic7XG4gICAgY29uc3QgeHM6IG51bWJlcltdID0gW107XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCAxMDsgKytpKSB7XG4gICAgICBjb25zdCByYW5kID0gbmV3IFVuaWZvcm1SYW5kb20obWluLCBtYXgsIGR0eXBlKTtcbiAgICAgIGNvbnN0IHggPSByYW5kLm5leHRWYWx1ZSgpO1xuICAgICAgeHMucHVzaCh4KTtcbiAgICB9XG4gICAgZXhwZWN0KE1hdGgubWluKC4uLnhzKSkudG9CZUdyZWF0ZXJUaGFuT3JFcXVhbChtaW4pO1xuICAgIGV4cGVjdChNYXRoLm1heCguLi54cykpLnRvQmVMZXNzVGhhbihtYXgpO1xuICB9KTtcblxuICBpdCgnaW50MzIsIG5vIHNlZWQnLCAoKSA9PiB7XG4gICAgY29uc3QgbWluID0gMTM7XG4gICAgY29uc3QgbWF4ID0gMzc7XG4gICAgY29uc3QgZHR5cGUgPSAnaW50MzInO1xuICAgIGNvbnN0IHhzOiBudW1iZXJbXSA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgMTA7ICsraSkge1xuICAgICAgY29uc3QgcmFuZCA9IG5ldyBVbmlmb3JtUmFuZG9tKG1pbiwgbWF4LCBkdHlwZSk7XG4gICAgICBjb25zdCB4ID0gcmFuZC5uZXh0VmFsdWUoKTtcbiAgICAgIGV4cGVjdChOdW1iZXIuaXNJbnRlZ2VyKHgpKS50b0VxdWFsKHRydWUpO1xuICAgICAgeHMucHVzaCh4KTtcbiAgICB9XG4gICAgZXhwZWN0KE1hdGgubWluKC4uLnhzKSkudG9CZUdyZWF0ZXJUaGFuT3JFcXVhbChtaW4pO1xuICAgIGV4cGVjdChNYXRoLm1heCguLi54cykpLnRvQmVMZXNzVGhhbk9yRXF1YWwobWF4KTtcbiAgfSk7XG5cbiAgaXQoJ3NlZWQgaXMgbnVtYmVyJywgKCkgPT4ge1xuICAgIGNvbnN0IG1pbiA9IC0xLjI7XG4gICAgY29uc3QgbWF4ID0gLTAuNDtcbiAgICBjb25zdCBkdHlwZSA9ICdmbG9hdDMyJztcbiAgICBjb25zdCBzZWVkID0gMTMzNztcbiAgICBjb25zdCB4czogbnVtYmVyW10gPSBbXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IDEwOyArK2kpIHtcbiAgICAgIGNvbnN0IHJhbmQgPSBuZXcgVW5pZm9ybVJhbmRvbShtaW4sIG1heCwgZHR5cGUsIHNlZWQpO1xuICAgICAgY29uc3QgeCA9IHJhbmQubmV4dFZhbHVlKCk7XG4gICAgICBleHBlY3QoeCkudG9CZUdyZWF0ZXJUaGFuT3JFcXVhbChtaW4pO1xuICAgICAgZXhwZWN0KHgpLnRvQmVMZXNzVGhhbihtYXgpO1xuICAgICAgeHMucHVzaCh4KTtcbiAgICB9XG4gICAgLy8gQXNzZXJ0IGRldGVybWluaXN0aWMgcmVzdWx0cy5cbiAgICBleHBlY3QoTWF0aC5taW4oLi4ueHMpKS50b0VxdWFsKE1hdGgubWF4KC4uLnhzKSk7XG4gIH0pO1xuXG4gIGl0KCdzZWVkID09PSBudWxsJywgKCkgPT4ge1xuICAgIGNvbnN0IG1pbiA9IDA7XG4gICAgY29uc3QgbWF4ID0gMTtcbiAgICBjb25zdCBkdHlwZSA9ICdmbG9hdDMyJztcbiAgICBjb25zdCBzZWVkOiBudW1iZXIgPSBudWxsO1xuICAgIGNvbnN0IHJhbmQgPSBuZXcgVW5pZm9ybVJhbmRvbShtaW4sIG1heCwgZHR5cGUsIHNlZWQpO1xuICAgIGNvbnN0IHggPSByYW5kLm5leHRWYWx1ZSgpO1xuICAgIGV4cGVjdCh4KS50b0JlR3JlYXRlclRoYW5PckVxdWFsKDApO1xuICAgIGV4cGVjdCh4KS50b0JlTGVzc1RoYW4oMSk7XG4gIH0pO1xuXG4gIGl0KCdzZWVkID09PSB1bmRlZmluZWQnLCAoKSA9PiB7XG4gICAgY29uc3QgbWluID0gMDtcbiAgICBjb25zdCBtYXggPSAxO1xuICAgIGNvbnN0IGR0eXBlID0gJ2Zsb2F0MzInO1xuICAgIGNvbnN0IHNlZWQ6IG51bWJlciA9IHVuZGVmaW5lZDtcbiAgICBjb25zdCByYW5kID0gbmV3IFVuaWZvcm1SYW5kb20obWluLCBtYXgsIGR0eXBlLCBzZWVkKTtcbiAgICBjb25zdCB4ID0gcmFuZC5uZXh0VmFsdWUoKTtcbiAgICBleHBlY3QoeCkudG9CZUdyZWF0ZXJUaGFuT3JFcXVhbCgwKTtcbiAgICBleHBlY3QoeCkudG9CZUxlc3NUaGFuKDEpO1xuICB9KTtcbn0pO1xuIl19