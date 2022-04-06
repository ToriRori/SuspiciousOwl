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
import { ENGINE } from './engine';
import { inferShape } from './tensor_util_env';
import { arraysEqual, encodeString, flatten, isString, isTypedArray } from './util';
const TEST_EPSILON_FLOAT32 = 1e-3;
export const TEST_EPSILON_FLOAT16 = 1e-1;
export function expectArraysClose(actual, expected, epsilon) {
    if (epsilon == null) {
        epsilon = testEpsilon();
    }
    return expectArraysPredicate(actual, expected, (a, b) => areClose(a, b, epsilon));
}
export function testEpsilon() {
    return ENGINE.backend.floatPrecision() === 32 ? TEST_EPSILON_FLOAT32 :
        TEST_EPSILON_FLOAT16;
}
function expectArraysPredicate(actual, expected, predicate) {
    let checkClassType = true;
    if (isTypedArray(actual) || isTypedArray(expected)) {
        checkClassType = false;
    }
    if (isTypedArray(actual) && isTypedArray(expected)) {
        checkClassType = true;
    }
    if (checkClassType) {
        const aType = actual.constructor.name;
        const bType = expected.constructor.name;
        if (aType !== bType) {
            throw new Error(`Arrays are of different type. Actual: ${aType}. ` +
                `Expected: ${bType}`);
        }
    }
    if (Array.isArray(actual) && Array.isArray(expected)) {
        const actualShape = inferShape(actual);
        const expectedShape = inferShape(expected);
        if (!arraysEqual(actualShape, expectedShape)) {
            throw new Error(`Arrays have different shapes. ` +
                `Actual: [${actualShape}]. Expected: [${expectedShape}]`);
        }
    }
    const actualFlat = isTypedArray(actual) ? actual : flatten(actual);
    const expectedFlat = isTypedArray(expected) ?
        expected :
        flatten(expected);
    if (actualFlat.length !== expectedFlat.length) {
        throw new Error(`Arrays have different lengths actual: ${actualFlat.length} vs ` +
            `expected: ${expectedFlat.length}.\n` +
            `Actual:   ${actualFlat}.\n` +
            `Expected: ${expectedFlat}.`);
    }
    for (let i = 0; i < expectedFlat.length; ++i) {
        const a = actualFlat[i];
        const e = expectedFlat[i];
        if (!predicate(a, e)) {
            throw new Error(`Arrays differ: actual[${i}] = ${a}, expected[${i}] = ${e}.\n` +
                `Actual:   ${actualFlat}.\n` +
                `Expected: ${expectedFlat}.`);
        }
    }
}
export function expectPromiseToFail(fn, done) {
    fn().then(() => done.fail(), () => done());
}
export function expectArraysEqual(actual, expected) {
    const exp = typeof expected === 'string' || typeof expected === 'number' ||
        typeof expected === 'boolean' ?
        [expected] :
        expected;
    if (isString(actual) || isString(actual[0]) ||
        isString(expected) || isString(expected[0])) {
        // tslint:disable-next-line: triple-equals
        return expectArraysPredicate(actual, exp, (a, b) => a == b);
    }
    return expectArraysPredicate(actual, expected, (a, b) => areClose(a, b, 0));
}
export function expectNumbersClose(a, e, epsilon) {
    if (epsilon == null) {
        epsilon = testEpsilon();
    }
    if (!areClose(a, e, epsilon)) {
        throw new Error(`Numbers differ: actual === ${a}, expected === ${e}`);
    }
}
function areClose(a, e, epsilon) {
    if (!isFinite(a) && !isFinite(e)) {
        return true;
    }
    if (isNaN(a) || isNaN(e) || Math.abs(a - e) > epsilon) {
        return false;
    }
    return true;
}
export function expectValuesInRange(actual, low, high) {
    for (let i = 0; i < actual.length; i++) {
        if (actual[i] < low || actual[i] > high) {
            throw new Error(`Value out of range:${actual[i]} low: ${low}, high: ${high}`);
        }
    }
}
export function expectArrayBuffersEqual(actual, expected) {
    // Safari does not like comparing ArrayBuffers directly. Wrapping in
    // a Float32Array solves this issue.
    const actualArray = new Float32Array(actual);
    const expectedArray = new Float32Array(expected);
    if (actualArray.length !== expectedArray.length) {
        throw new Error('Expected ArrayBuffer to be of length ' +
            `${expectedArray.length}, but it was ${actualArray.length}`);
    }
    for (let i = 0; i < expectedArray.length; i++) {
        if (actualArray[i] !== expectedArray[i]) {
            throw new Error(`Expected ArrayBuffer value at ${i} to be ` +
                `${expectedArray[i]} but got ${actualArray[i]} instead`);
        }
    }
}
/** Encodes strings into utf-8 bytes. */
export function encodeStrings(a) {
    for (let i = 0; i < a.length; i++) {
        const val = a[i];
        if (Array.isArray(val)) {
            encodeStrings(val);
        }
        else {
            a[i] = encodeString(val);
        }
    }
    return a;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidGVzdF91dGlsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy90ZXN0X3V0aWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUNoQyxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFFN0MsT0FBTyxFQUFDLFdBQVcsRUFBRSxZQUFZLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBRSxZQUFZLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFFbEYsTUFBTSxvQkFBb0IsR0FBRyxJQUFJLENBQUM7QUFDbEMsTUFBTSxDQUFDLE1BQU0sb0JBQW9CLEdBQUcsSUFBSSxDQUFDO0FBRXpDLE1BQU0sVUFBVSxpQkFBaUIsQ0FDN0IsTUFBZ0QsRUFDaEQsUUFBa0QsRUFBRSxPQUFnQjtJQUN0RSxJQUFJLE9BQU8sSUFBSSxJQUFJLEVBQUU7UUFDbkIsT0FBTyxHQUFHLFdBQVcsRUFBRSxDQUFDO0tBQ3pCO0lBQ0QsT0FBTyxxQkFBcUIsQ0FDeEIsTUFBTSxFQUFFLFFBQVEsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFXLEVBQUUsQ0FBVyxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUM7QUFDL0UsQ0FBQztBQUVELE1BQU0sVUFBVSxXQUFXO0lBQ3pCLE9BQU8sTUFBTSxDQUFDLE9BQU8sQ0FBQyxjQUFjLEVBQUUsS0FBSyxFQUFFLENBQUMsQ0FBQyxDQUFDLG9CQUFvQixDQUFDLENBQUM7UUFDdEIsb0JBQW9CLENBQUM7QUFDdkUsQ0FBQztBQUVELFNBQVMscUJBQXFCLENBQzFCLE1BQWtCLEVBQUUsUUFBb0IsRUFDeEMsU0FBb0M7SUFDdEMsSUFBSSxjQUFjLEdBQUcsSUFBSSxDQUFDO0lBQzFCLElBQUksWUFBWSxDQUFDLE1BQU0s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QUU7WUFDdkMsTUFBTSxJQUFJLEtBQUssQ0FDWCxpQ0FBaUMsQ0FBQyxTQUFTO2dCQUMzQyxHQUFHLGFBQWEsQ0FBQyxDQUFDLENBQUMsWUFBWSxXQUFXLENBQUMsQ0FBQyxDQUFDLFVBQVUsQ0FBQyxDQUFDO1NBQzlEO0tBQ0Y7QUFDSCxDQUFDO0FBRUQsd0NBQXdDO0FBQ3hDLE1BQU0sVUFBVSxhQUFhLENBQUMsQ0FBcUI7SUFFakQsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFJLENBQWUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUU7UUFDaEQsTUFBTSxHQUFHLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2pCLElBQUksS0FBSyxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUMsRUFBRTtZQUN0QixhQUFhLENBQUMsR0FBRyxDQUFDLENBQUM7U0FDcEI7YUFBTTtZQUNMLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxZQUFZLENBQUMsR0FBYSxDQUFDLENBQUM7U0FDcEM7S0FDRjtJQUNELE9BQU8sQ0FBK0IsQ0FBQztBQUN6QyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi9lbmdpbmUnO1xuaW1wb3J0IHtpbmZlclNoYXBlfSBmcm9tICcuL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1JlY3Vyc2l2ZUFycmF5LCBUZW5zb3JMaWtlLCBUeXBlZEFycmF5fSBmcm9tICcuL3R5cGVzJztcbmltcG9ydCB7YXJyYXlzRXF1YWwsIGVuY29kZVN0cmluZywgZmxhdHRlbiwgaXNTdHJpbmcsIGlzVHlwZWRBcnJheX0gZnJvbSAnLi91dGlsJztcblxuY29uc3QgVEVTVF9FUFNJTE9OX0ZMT0FUMzIgPSAxZS0zO1xuZXhwb3J0IGNvbnN0IFRFU1RfRVBTSUxPTl9GTE9BVDE2ID0gMWUtMTtcblxuZXhwb3J0IGZ1bmN0aW9uIGV4cGVjdEFycmF5c0Nsb3NlKFxuICAgIGFjdHVhbDogVHlwZWRBcnJheXxudW1iZXJ8UmVjdXJzaXZlQXJyYXk8bnVtYmVyPixcbiAgICBleHBlY3RlZDogVHlwZWRBcnJheXxudW1iZXJ8UmVjdXJzaXZlQXJyYXk8bnVtYmVyPiwgZXBzaWxvbj86IG51bWJlcikge1xuICBpZiAoZXBzaWxvbiA9PSBudWxsKSB7XG4gICAgZXBzaWxvbiA9IHRlc3RFcHNpbG9uKCk7XG4gIH1cbiAgcmV0dXJuIGV4cGVjdEFycmF5c1ByZWRpY2F0ZShcbiAgICAgIGFjdHVhbCwgZXhwZWN0ZWQsIChhLCBiKSA9PiBhcmVDbG9zZShhIGFzIG51bWJlciwgYiBhcyBudW1iZXIsIGVwc2lsb24pKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHRlc3RFcHNpbG9uKCkge1xuICByZXR1cm4gRU5HSU5FLmJhY2tlbmQuZmxvYXRQcmVjaXNpb24oKSA9PT0gMzIgPyBURVNUX0VQU0lMT05fRkxPQVQzMiA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFRFU1RfRVBTSUxPTl9GTE9BVDE2O1xufVxuXG5mdW5jdGlvbiBleHBlY3RBcnJheXNQcmVkaWNhdGUoXG4gICAgYWN0dWFsOiBUZW5zb3JMaWtlLCBleHBlY3RlZDogVGVuc29yTGlrZSxcbiAgICBwcmVkaWNhdGU6IChhOiB7fSwgYjoge30pID0+IGJvb2xlYW4pIHtcbiAgbGV0IGNoZWNrQ2xhc3NUeXBlID0gdHJ1ZTtcbiAgaWYgKGlzVHlwZWRBcnJheShhY3R1YWwpIHx8IGlzVHlwZWRBcnJheShleHBlY3RlZCkpIHtcbiAgICBjaGVja0NsYXNzVHlwZSA9IGZhbHNlO1xuICB9XG4gIGlmIChpc1R5cGVkQXJyYXkoYWN0dWFsKSAmJiBpc1R5cGVkQXJyYXkoZXhwZWN0ZWQpKSB7XG4gICAgY2hlY2tDbGFzc1R5cGUgPSB0cnVlO1xuICB9XG4gIGlmIChjaGVja0NsYXNzVHlwZSkge1xuICAgIGNvbnN0IGFUeXBlID0gYWN0dWFsLmNvbnN0cnVjdG9yLm5hbWU7XG4gICAgY29uc3QgYlR5cGUgPSBleHBlY3RlZC5jb25zdHJ1Y3Rvci5uYW1lO1xuXG4gICAgaWYgKGFUeXBlICE9PSBiVHlwZSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBBcnJheXMgYXJlIG9mIGRpZmZlcmVudCB0eXBlLiBBY3R1YWw6ICR7YVR5cGV9LiBgICtcbiAgICAgICAgICBgRXhwZWN0ZWQ6ICR7YlR5cGV9YCk7XG4gICAgfVxuICB9XG5cbiAgaWYgKEFycmF5LmlzQXJyYXkoYWN0dWFsKSAmJiBBcnJheS5pc0FycmF5KGV4cGVjdGVkKSkge1xuICAgIGNvbnN0IGFjdHVhbFNoYXBlID0gaW5mZXJTaGFwZShhY3R1YWwpO1xuICAgIGNvbnN0IGV4cGVjdGVkU2hhcGUgPSBpbmZlclNoYXBlKGV4cGVjdGVkKTtcbiAgICBpZiAoIWFycmF5c0VxdWFsKGFjdHVhbFNoYXBlLCBleHBlY3RlZFNoYXBlKSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBBcnJheXMgaGF2ZSBkaWZmZXJlbnQgc2hhcGVzLiBgICtcbiAgICAgICAgICBgQWN0dWFsOiBbJHthY3R1YWxTaGFwZX1dLiBFeHBlY3RlZDogWyR7ZXhwZWN0ZWRTaGFwZX1dYCk7XG4gICAgfVxuICB9XG5cbiAgY29uc3QgYWN0dWFsRmxhdCA9XG4gICAgICBpc1R5cGVkQXJyYXkoYWN0dWFsKSA/IGFjdHVhbCA6IGZsYXR0ZW4oYWN0dWFsIGFzIFJlY3Vyc2l2ZUFycmF5PG51bWJlcj4pO1xuICBjb25zdCBleHBlY3RlZEZsYXQgPSBpc1R5cGVkQXJyYXkoZXhwZWN0ZWQpID9cbiAgICAgIGV4cGVjdGVkIDpcbiAgICAgIGZsYXR0ZW4oZXhwZWN0ZWQgYXMgUmVjdXJzaXZlQXJyYXk8bnVtYmVyPik7XG5cbiAgaWYgKGFjdHVhbEZsYXQubGVuZ3RoICE9PSBleHBlY3RlZEZsYXQubGVuZ3RoKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgQXJyYXlzIGhhdmUgZGlmZmVyZW50IGxlbmd0aHMgYWN0dWFsOiAke2FjdHVhbEZsYXQubGVuZ3RofSB2cyBgICtcbiAgICAgICAgYGV4cGVjdGVkOiAke2V4cGVjdGVkRmxhdC5sZW5ndGh9LlxcbmAgK1xuICAgICAgICBgQWN0dWFsOiAgICR7YWN0dWFsRmxhdH0uXFxuYCArXG4gICAgICAgIGBFeHBlY3RlZDogJHtleHBlY3RlZEZsYXR9LmApO1xuICB9XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgZXhwZWN0ZWRGbGF0Lmxlbmd0aDsgKytpKSB7XG4gICAgY29uc3QgYSA9IGFjdHVhbEZsYXRbaV07XG4gICAgY29uc3QgZSA9IGV4cGVjdGVkRmxhdFtpXTtcblxuICAgIGlmICghcHJlZGljYXRlKGEsIGUpKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYEFycmF5cyBkaWZmZXI6IGFjdHVhbFske2l9XSA9ICR7YX0sIGV4cGVjdGVkWyR7aX1dID0gJHtlfS5cXG5gICtcbiAgICAgICAgICBgQWN0dWFsOiAgICR7YWN0dWFsRmxhdH0uXFxuYCArXG4gICAgICAgICAgYEV4cGVjdGVkOiAke2V4cGVjdGVkRmxhdH0uYCk7XG4gICAgfVxuICB9XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgRG9uZUZuIHtcbiAgKCk6IHZvaWQ7XG4gIGZhaWw6IChtZXNzYWdlPzogRXJyb3J8c3RyaW5nKSA9PiB2b2lkO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZXhwZWN0UHJvbWlzZVRvRmFpbChmbjogKCkgPT4gUHJvbWlzZTx7fT4sIGRvbmU6IERvbmVGbik6IHZvaWQge1xuICBmbigpLnRoZW4oKCkgPT4gZG9uZS5mYWlsKCksICgpID0+IGRvbmUoKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBleHBlY3RBcnJheXNFcXVhbChhY3R1YWw6IFRlbnNvckxpa2UsIGV4cGVjdGVkOiBUZW5zb3JMaWtlKSB7XG4gIGNvbnN0IGV4cCA9IHR5cGVvZiBleHBlY3RlZCA9PT0gJ3N0cmluZycgfHwgdHlwZW9mIGV4cGVjdGVkID09PSAnbnVtYmVyJyB8fFxuICAgICAgICAgIHR5cGVvZiBleHBlY3RlZCA9PT0gJ2Jvb2xlYW4nID9cbiAgICAgIFtleHBlY3RlZF0gYXMgbnVtYmVyW10gOlxuICAgICAgZXhwZWN0ZWQgYXMgbnVtYmVyW107XG4gIGlmIChpc1N0cmluZyhhY3R1YWwpIHx8IGlzU3RyaW5nKChhY3R1YWwgYXMgc3RyaW5nW10pWzBdKSB8fFxuICAgICAgaXNTdHJpbmcoZXhwZWN0ZWQpIHx8IGlzU3RyaW5nKChleHBlY3RlZCBhcyBzdHJpbmdbXSlbMF0pKSB7XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiB0cmlwbGUtZXF1YWxzXG4gICAgcmV0dXJuIGV4cGVjdEFycmF5c1ByZWRpY2F0ZShhY3R1YWwsIGV4cCwgKGEsIGIpID0+IGEgPT0gYik7XG4gIH1cbiAgcmV0dXJuIGV4cGVjdEFycmF5c1ByZWRpY2F0ZShcbiAgICAgIGFjdHVhbCwgZXhwZWN0ZWQsIChhLCBiKSA9PiBhcmVDbG9zZShhIGFzIG51bWJlciwgYiBhcyBudW1iZXIsIDApKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGV4cGVjdE51bWJlcnNDbG9zZShhOiBudW1iZXIsIGU6IG51bWJlciwgZXBzaWxvbj86IG51bWJlcikge1xuICBpZiAoZXBzaWxvbiA9PSBudWxsKSB7XG4gICAgZXBzaWxvbiA9IHRlc3RFcHNpbG9uKCk7XG4gIH1cbiAgaWYgKCFhcmVDbG9zZShhLCBlLCBlcHNpbG9uKSkge1xuICAgIHRocm93IG5ldyBFcnJvcihgTnVtYmVycyBkaWZmZXI6IGFjdHVhbCA9PT0gJHthfSwgZXhwZWN0ZWQgPT09ICR7ZX1gKTtcbiAgfVxufVxuXG5mdW5jdGlvbiBhcmVDbG9zZShhOiBudW1iZXIsIGU6IG51bWJlciwgZXBzaWxvbjogbnVtYmVyKTogYm9vbGVhbiB7XG4gIGlmICghaXNGaW5pdGUoYSkgJiYgIWlzRmluaXRlKGUpKSB7XG4gICAgcmV0dXJuIHRydWU7XG4gIH1cbiAgaWYgKGlzTmFOKGEpIHx8IGlzTmFOKGUpIHx8IE1hdGguYWJzKGEgLSBlKSA+IGVwc2lsb24pIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbiAgcmV0dXJuIHRydWU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBleHBlY3RWYWx1ZXNJblJhbmdlKFxuICAgIGFjdHVhbDogVHlwZWRBcnJheXxudW1iZXJbXSwgbG93OiBudW1iZXIsIGhpZ2g6IG51bWJlcikge1xuICBmb3IgKGxldCBpID0gMDsgaSA8IGFjdHVhbC5sZW5ndGg7IGkrKykge1xuICAgIGlmIChhY3R1YWxbaV0gPCBsb3cgfHwgYWN0dWFsW2ldID4gaGlnaCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBWYWx1ZSBvdXQgb2YgcmFuZ2U6JHthY3R1YWxbaV19IGxvdzogJHtsb3d9LCBoaWdoOiAke2hpZ2h9YCk7XG4gICAgfVxuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBleHBlY3RBcnJheUJ1ZmZlcnNFcXVhbChcbiAgICBhY3R1YWw6IEFycmF5QnVmZmVyLCBleHBlY3RlZDogQXJyYXlCdWZmZXIpIHtcbiAgLy8gU2FmYXJpIGRvZXMgbm90IGxpa2UgY29tcGFyaW5nIEFycmF5QnVmZmVycyBkaXJlY3RseS4gV3JhcHBpbmcgaW5cbiAgLy8gYSBGbG9hdDMyQXJyYXkgc29sdmVzIHRoaXMgaXNzdWUuXG4gIGNvbnN0IGFjdHVhbEFycmF5ID0gbmV3IEZsb2F0MzJBcnJheShhY3R1YWwpO1xuICBjb25zdCBleHBlY3RlZEFycmF5ID0gbmV3IEZsb2F0MzJBcnJheShleHBlY3RlZCk7XG4gIGlmIChhY3R1YWxBcnJheS5sZW5ndGggIT09IGV4cGVjdGVkQXJyYXkubGVuZ3RoKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnRXhwZWN0ZWQgQXJyYXlCdWZmZXIgdG8gYmUgb2YgbGVuZ3RoICcgK1xuICAgICAgICBgJHtleHBlY3RlZEFycmF5Lmxlbmd0aH0sIGJ1dCBpdCB3YXMgJHthY3R1YWxBcnJheS5sZW5ndGh9YCk7XG4gIH1cblxuICBmb3IgKGxldCBpID0gMDsgaSA8IGV4cGVjdGVkQXJyYXkubGVuZ3RoOyBpKyspIHtcbiAgICBpZiAoYWN0dWFsQXJyYXlbaV0gIT09IGV4cGVjdGVkQXJyYXlbaV0pIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgRXhwZWN0ZWQgQXJyYXlCdWZmZXIgdmFsdWUgYXQgJHtpfSB0byBiZSBgICtcbiAgICAgICAgICBgJHtleHBlY3RlZEFycmF5W2ldfSBidXQgZ290ICR7YWN0dWFsQXJyYXlbaV19IGluc3RlYWRgKTtcbiAgICB9XG4gIH1cbn1cblxuLyoqIEVuY29kZXMgc3RyaW5ncyBpbnRvIHV0Zi04IGJ5dGVzLiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGVuY29kZVN0cmluZ3MoYTogUmVjdXJzaXZlQXJyYXk8e30+KTpcbiAgICBSZWN1cnNpdmVBcnJheTxVaW50OEFycmF5PiB7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgKGEgYXMgQXJyYXk8e30+KS5sZW5ndGg7IGkrKykge1xuICAgIGNvbnN0IHZhbCA9IGFbaV07XG4gICAgaWYgKEFycmF5LmlzQXJyYXkodmFsKSkge1xuICAgICAgZW5jb2RlU3RyaW5ncyh2YWwpO1xuICAgIH0gZWxzZSB7XG4gICAgICBhW2ldID0gZW5jb2RlU3RyaW5nKHZhbCBhcyBzdHJpbmcpO1xuICAgIH1cbiAgfVxuICByZXR1cm4gYSBhcyBSZWN1cnNpdmVBcnJheTxVaW50OEFycmF5Pjtcbn1cbiJdfQ==