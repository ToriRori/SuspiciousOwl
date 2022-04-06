/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
// base.ts is tfjs-core without auto registration of things like flags,
// gradients, chained ops or the opHandler. See base_side_effects.ts for parts
// tfjs core that are required side effects.
/**
 * @fileoverview
 * @suppress {partialAlias} Optimization disabled due to passing the module
 * object into a function below:
 *
 *   import * as ops from './ops/ops';
 *   setOpHandler(ops);
 */
// Serialization.
import * as io from './io/io';
import * as math from './math';
import * as broadcast_util from './ops/broadcast_util';
import * as browser from './ops/browser';
import * as gather_util from './ops/gather_nd_util';
import * as scatter_util from './ops/scatter_nd_util';
import * as slice_util from './ops/slice_util';
import * as serialization from './serialization';
import * as tensor_util from './tensor_util';
import * as test_util from './test_util';
import * as util from './util';
import { version } from './version';
export { AdadeltaOptimizer } from './optimizers/adadelta_optimizer';
export { AdagradOptimizer } from './optimizers/adagrad_optimizer';
export { AdamOptimizer } from './optimizers/adam_optimizer';
export { AdamaxOptimizer } from './optimizers/adamax_optimizer';
export { MomentumOptimizer } from './optimizers/momentum_optimizer';
export { Optimizer } from './optimizers/optimizer';
// Optimizers.
export { OptimizerConstructors } from './optimizers/optimizer_constructors';
export { RMSPropOptimizer } from './optimizers/rmsprop_optimizer';
export { SGDOptimizer } from './optimizers/sgd_optimizer';
export { Tensor, TensorBuffer, Variable } from './tensor';
export { Rank, sumOutType, upcastType } from './types';
export * from './ops/ops';
export { Reduction } from './ops/loss_ops_utils';
export * from './train';
export * from './globals';
export * from './kernel_registry';
export { customGrad, grad, grads, valueAndGrad, valueAndGrads, variableGrads } from './gradients';
export { Environment, env, ENV } from './environment';
export { version as version_core };
// Top-level method exports.
export { nextFrame } from './browser_util';
// Second level exports.
import * as backend_util from './backends/backend_util';
import * as device_util from './device_util';
export { browser, io, math, serialization, test_util, util, backend_util, broadcast_util, tensor_util, slice_util, gather_util, scatter_util, device_util };
import * as kernel_impls from './backends/kernel_impls';
export { kernel_impls };
// Backend specific.
export { KernelBackend, DataStorage } from './backends/backend';
// Export all kernel names / info.
export * from './kernel_names';
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFzZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvYmFzZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCx1RUFBdUU7QUFDdkUsOEVBQThFO0FBQzlFLDRDQUE0QztBQUU1Qzs7Ozs7OztHQU9HO0FBRUgsaUJBQWlCO0FBQ2pCLE9BQU8sS0FBSyxFQUFFLE1BQU0sU0FBUyxDQUFDO0FBQzlCLE9BQU8sS0FBSyxJQUFJLE1BQU0sUUFBUSxDQUFDO0FBQy9CLE9BQU8sS0FBSyxjQUFjLE1BQU0sc0JBQXNCLENBQUM7QUFDdkQsT0FBTyxLQUFLLE9BQU8sTUFBTSxlQUFlLENBQUM7QUFDekMsT0FBTyxLQUFLLFdBQVcsTUFBTSxzQkFBc0IsQ0FBQztBQUNwRCxPQUFPLEtBQUssWUFBWSxNQUFNLHVCQUF1QixDQUFDO0FBQ3RELE9BQU8sS0FBSyxVQUFVLE1BQU0sa0JBQWtCLENBQUM7QUFDL0MsT0FBTyxLQUFLLGFBQWEsTUFBTSxpQkFBaUIsQ0FBQztBQUNqRCxPQUFPLEtBQUssV0FBVyxNQUFNLGVBQWUsQ0FBQztBQUM3QyxPQUFPLEtBQUssU0FBUyxNQUFNLGFBQWEsQ0FBQztBQUN6QyxPQUFPLEtBQUssSUFBSSxNQUFNLFFBQVEsQ0FBQztBQUMvQixPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBR2xDLE9BQU8sRUFBQyxpQkFBaUIsRUFBQyxNQUFNLGlDQUFpQyxDQUFDO0FBQ2xFLE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLGdDQUFnQyxDQUFDO0FBQ2hFLE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSw2QkFBNkIsQ0FBQztBQUMxRCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sK0JBQStCLENBQUM7QUFDOUQsT0FBTyxFQUFDLGlCQUFpQixFQUFDLE1BQU0saUNBQWlDLENBQUM7QUFDbEUsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBQ2pELGNBQWM7QUFDZCxPQUFPLEVBQUMscUJBQXFCLEVBQUMsTUFBTSxxQ0FBcUMsQ0FBQztBQUMxRSxPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxnQ0FBZ0MsQ0FBQztBQUNoRSxPQUFPLEVBQUMsWUFBWSxFQUFDLE1BQU0sNEJBQTRCLENBQUM7QUFDeEQsT0FBTyxFQUEwRCxNQUFNLEVBQW9ELFlBQVksRUFBRSxRQUFRLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFFbkssT0FBTyxFQUErRSxJQUFJLEVBQXdDLFVBQVUsRUFBMEIsVUFBVSxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBRWpNLGNBQWMsV0FBVyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxTQUFTLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUUvQyxjQUFjLFNBQVMsQ0FBQztBQUN4QixjQUFjLFdBQVcsQ0FBQztBQUMxQixjQUFjLG1CQUFtQixDQUFDO0FBQ2xDLE9BQU8sRUFBQyxVQUFVLEVBQUUsSUFBSSxFQUFFLEtBQUssRUFBRSxZQUFZLEVBQUUsYUFBYSxFQUFFLGFBQWEsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUdoRyxPQUFPLEVBQUMsV0FBVyxFQUFFLEdBQUcsRUFBRSxHQUFHLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFHcEQsT0FBTyxFQUFDLE9BQU8sSUFBSSxZQUFZLEVBQUMsQ0FBQztBQUVqQyw0QkFBNEI7QUFDNUIsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBRXpDLHdCQUF3QjtBQUN4QixPQUFPLEtBQUssWUFBWSxNQUFNLHlCQUF5QixDQUFDO0FBQ3hELE9BQU8sS0FBSyxXQUFXLE1BQU0sZUFBZSxDQUFDO0FBQzdDLE9BQU8sRUFDTCxPQUFPLEVBQ1AsRUFBRSxFQUNGLElBQUksRUFDSixhQUFhLEVBQ2IsU0FBUyxFQUNULElBQUksRUFDSixZQUFZLEVBQ1osY0FBYyxFQUNkLFdBQVcsRUFDWCxVQUFVLEVBQ1YsV0FBVyxFQUNYLFlBQVksRUFDWixXQUFXLEVBQ1osQ0FBQztBQUVGLE9BQU8sS0FBSyxZQUFZLE1BQU0seUJBQXlCLENBQUM7QUFDeEQsT0FBTyxFQUFDLFlBQVksRUFBQyxDQUFDO0FBQ3RCLG9CQUFvQjtBQUNwQixPQUFPLEVBQUMsYUFBYSxFQUFnQyxXQUFXLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUU1RixrQ0FBa0M7QUFDbEMsY0FBYyxnQkFBZ0IsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLy8gYmFzZS50cyBpcyB0ZmpzLWNvcmUgd2l0aG91dCBhdXRvIHJlZ2lzdHJhdGlvbiBvZiB0aGluZ3MgbGlrZSBmbGFncyxcbi8vIGdyYWRpZW50cywgY2hhaW5lZCBvcHMgb3IgdGhlIG9wSGFuZGxlci4gU2VlIGJhc2Vfc2lkZV9lZmZlY3RzLnRzIGZvciBwYXJ0c1xuLy8gdGZqcyBjb3JlIHRoYXQgYXJlIHJlcXVpcmVkIHNpZGUgZWZmZWN0cy5cblxuLyoqXG4gKiBAZmlsZW92ZXJ2aWV3XG4gKiBAc3VwcHJlc3Mge3BhcnRpYWxBbGlhc30gT3B0aW1pemF0aW9uIGRpc2FibGVkIGR1ZSB0byBwYXNzaW5nIHRoZSBtb2R1bGVcbiAqIG9iamVjdCBpbnRvIGEgZnVuY3Rpb24gYmVsb3c6XG4gKlxuICogICBpbXBvcnQgKiBhcyBvcHMgZnJvbSAnLi9vcHMvb3BzJztcbiAqICAgc2V0T3BIYW5kbGVyKG9wcyk7XG4gKi9cblxuLy8gU2VyaWFsaXphdGlvbi5cbmltcG9ydCAqIGFzIGlvIGZyb20gJy4vaW8vaW8nO1xuaW1wb3J0ICogYXMgbWF0aCBmcm9tICcuL21hdGgnO1xuaW1wb3J0ICogYXMgYnJvYWRjYXN0X3V0aWwgZnJvbSAnLi9vcHMvYnJvYWRjYXN0X3V0aWwnO1xuaW1wb3J0ICogYXMgYnJvd3NlciBmcm9tICcuL29wcy9icm93c2VyJztcbmltcG9ydCAqIGFzIGdhdGhlcl91dGlsIGZyb20gJy4vb3BzL2dhdGhlcl9uZF91dGlsJztcbmltcG9ydCAqIGFzIHNjYXR0ZXJfdXRpbCBmcm9tICcuL29wcy9zY2F0dGVyX25kX3V0aWwnO1xuaW1wb3J0ICogYXMgc2xpY2VfdXRpbCBmcm9tICcuL29wcy9zbGljZV91dGlsJztcbmltcG9ydCAqIGFzIHNlcmlhbGl6YXRpb24gZnJvbSAnLi9zZXJpYWxpemF0aW9uJztcbmltcG9ydCAqIGFzIHRlbnNvcl91dGlsIGZyb20gJy4vdGVuc29yX3V0aWwnO1xuaW1wb3J0ICogYXMgdGVzdF91dGlsIGZyb20gJy4vdGVzdF91dGlsJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi91dGlsJztcbmltcG9ydCB7dmVyc2lvbn0gZnJvbSAnLi92ZXJzaW9uJztcblxuZXhwb3J0IHtJbmZlcmVuY2VNb2RlbCwgTWV0YUdyYXBoLCBNZXRhR3JhcGhJbmZvLCBNb2RlbFByZWRpY3RDb25maWcsIE1vZGVsVGVuc29ySW5mbywgU2F2ZWRNb2RlbFRlbnNvckluZm8sIFNpZ25hdHVyZURlZiwgU2lnbmF0dXJlRGVmRW50cnksIFNpZ25hdHVyZURlZkluZm99IGZyb20gJy4vbW9kZWxfdHlwZXMnO1xuZXhwb3J0IHtBZGFkZWx0YU9wdGltaXplcn0gZnJvbSAnLi9vcHRpbWl6ZXJzL2FkYWRlbHRhX29wdGltaXplcic7XG5leHBvcnQge0FkYWdyYWRPcHRpbWl6ZXJ9IGZyb20gJy4vb3B0aW1pemVycy9hZGFncmFkX29wdGltaXplcic7XG5leHBvcnQge0FkYW1PcHRpbWl6ZXJ9IGZyb20gJy4vb3B0aW1pemVycy9hZGFtX29wdGltaXplcic7XG5leHBvcnQge0FkYW1heE9wdGltaXplcn0gZnJvbSAnLi9vcHRpbWl6ZXJzL2FkYW1heF9vcHRpbWl6ZXInO1xuZXhwb3J0IHtNb21lbnR1bU9wdGltaXplcn0gZnJvbSAnLi9vcHRpbWl6ZXJzL21vbWVudHVtX29wdGltaXplcic7XG5leHBvcnQge09wdGltaXplcn0gZnJvbSAnLi9vcHRpbWl6ZXJzL29wdGltaXplcic7XG4vLyBPcHRpbWl6ZXJzLlxuZXhwb3J0IHtPcHRpbWl6ZXJDb25zdHJ1Y3RvcnN9IGZyb20gJy4vb3B0aW1pemVycy9vcHRpbWl6ZXJfY29uc3RydWN0b3JzJztcbmV4cG9ydCB7Uk1TUHJvcE9wdGltaXplcn0gZnJvbSAnLi9vcHRpbWl6ZXJzL3Jtc3Byb3Bfb3B0aW1pemVyJztcbmV4cG9ydCB7U0dET3B0aW1pemVyfSBmcm9tICcuL29wdGltaXplcnMvc2dkX29wdGltaXplcic7XG5leHBvcnQge0RhdGFUb0dQVU9wdGlvbnMsIERhdGFUb0dQVVdlYkdMT3B0aW9uLCBHUFVEYXRhLCBTY2FsYXIsIFRlbnNvciwgVGVuc29yMUQsIFRlbnNvcjJELCBUZW5zb3IzRCwgVGVuc29yNEQsIFRlbnNvcjVELCBUZW5zb3JCdWZmZXIsIFZhcmlhYmxlfSBmcm9tICcuL3RlbnNvcic7XG5leHBvcnQge0dyYWRTYXZlRnVuYywgTmFtZWRUZW5zb3JNYXAsIFRlbnNvckNvbnRhaW5lciwgVGVuc29yQ29udGFpbmVyQXJyYXksIFRlbnNvckNvbnRhaW5lck9iamVjdH0gZnJvbSAnLi90ZW5zb3JfdHlwZXMnO1xuZXhwb3J0IHtCYWNrZW5kVmFsdWVzLCBEYXRhVHlwZSwgRGF0YVR5cGVNYXAsIERhdGFWYWx1ZXMsIE51bWVyaWNEYXRhVHlwZSwgUGl4ZWxEYXRhLCBSYW5rLCBSZWN1cnNpdmVBcnJheSwgU2NhbGFyTGlrZSwgU2hhcGVNYXAsIHN1bU91dFR5cGUsIFRlbnNvckxpa2UsIFR5cGVkQXJyYXksIHVwY2FzdFR5cGV9IGZyb20gJy4vdHlwZXMnO1xuXG5leHBvcnQgKiBmcm9tICcuL29wcy9vcHMnO1xuZXhwb3J0IHtSZWR1Y3Rpb259IGZyb20gJy4vb3BzL2xvc3Nfb3BzX3V0aWxzJztcblxuZXhwb3J0ICogZnJvbSAnLi90cmFpbic7XG5leHBvcnQgKiBmcm9tICcuL2dsb2JhbHMnO1xuZXhwb3J0ICogZnJvbSAnLi9rZXJuZWxfcmVnaXN0cnknO1xuZXhwb3J0IHtjdXN0b21HcmFkLCBncmFkLCBncmFkcywgdmFsdWVBbmRHcmFkLCB2YWx1ZUFuZEdyYWRzLCB2YXJpYWJsZUdyYWRzfSBmcm9tICcuL2dyYWRpZW50cyc7XG5cbmV4cG9ydCB7VGltaW5nSW5mbywgTWVtb3J5SW5mbywgRm9yd2FyZEZ1bmN9IGZyb20gJy4vZW5naW5lJztcbmV4cG9ydCB7RW52aXJvbm1lbnQsIGVudiwgRU5WfSBmcm9tICcuL2Vudmlyb25tZW50JztcbmV4cG9ydCB7UGxhdGZvcm19IGZyb20gJy4vcGxhdGZvcm1zL3BsYXRmb3JtJztcblxuZXhwb3J0IHt2ZXJzaW9uIGFzIHZlcnNpb25fY29yZX07XG5cbi8vIFRvcC1sZXZlbCBtZXRob2QgZXhwb3J0cy5cbmV4cG9ydCB7bmV4dEZyYW1lfSBmcm9tICcuL2Jyb3dzZXJfdXRpbCc7XG5cbi8vIFNlY29uZCBsZXZlbCBleHBvcnRzLlxuaW1wb3J0ICogYXMgYmFja2VuZF91dGlsIGZyb20gJy4vYmFja2VuZHMvYmFja2VuZF91dGlsJztcbmltcG9ydCAqIGFzIGRldmljZV91dGlsIGZyb20gJy4vZGV2aWNlX3V0aWwnO1xuZXhwb3J0IHtcbiAgYnJvd3NlcixcbiAgaW8sXG4gIG1hdGgsXG4gIHNlcmlhbGl6YXRpb24sXG4gIHRlc3RfdXRpbCxcbiAgdXRpbCxcbiAgYmFja2VuZF91dGlsLFxuICBicm9hZGNhc3RfdXRpbCxcbiAgdGVuc29yX3V0aWwsXG4gIHNsaWNlX3V0aWwsXG4gIGdhdGhlcl91dGlsLFxuICBzY2F0dGVyX3V0aWwsXG4gIGRldmljZV91dGlsXG59O1xuXG5pbXBvcnQgKiBhcyBrZXJuZWxfaW1wbHMgZnJvbSAnLi9iYWNrZW5kcy9rZXJuZWxfaW1wbHMnO1xuZXhwb3J0IHtrZXJuZWxfaW1wbHN9O1xuLy8gQmFja2VuZCBzcGVjaWZpYy5cbmV4cG9ydCB7S2VybmVsQmFja2VuZCwgQmFja2VuZFRpbWluZ0luZm8sIERhdGFNb3ZlciwgRGF0YVN0b3JhZ2V9IGZyb20gJy4vYmFja2VuZHMvYmFja2VuZCc7XG5cbi8vIEV4cG9ydCBhbGwga2VybmVsIG5hbWVzIC8gaW5mby5cbmV4cG9ydCAqIGZyb20gJy4va2VybmVsX25hbWVzJztcbiJdfQ==