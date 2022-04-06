import { assert } from '../util';
import { op } from './operation';
import { pad } from './pad';
/**
 * Pads a `tf.Tensor4D` with a given value and paddings. See `pad` for details.
 */
function pad4d_(x, paddings, constantValue = 0) {
    assert(paddings.length === 4 && paddings[0].length === 2 &&
        paddings[1].length === 2 && paddings[2].length === 2 &&
        paddings[3].length === 2, () => 'Invalid number of paddings. Must be length of 2 each.');
    return pad(x, paddings, constantValue);
}
export const pad4d = op({ pad4d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicGFkNGQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9wYWQ0ZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFrQkEsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUMvQixPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFFMUI7O0dBRUc7QUFDSCxTQUFTLE1BQU0sQ0FDWCxDQUFzQixFQUN0QixRQUdLLEVBQ0wsYUFBYSxHQUFHLENBQUM7SUFDbkIsTUFBTSxDQUNGLFFBQVEsQ0FBQyxNQUFNLEtBQUssQ0FBQyxJQUFJLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxNQUFNLEtBQUssQ0FBQztRQUM3QyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTSxLQUFLLENBQUMsSUFBSSxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTSxLQUFLLENBQUM7UUFDcEQsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQzVCLEdBQUcsRUFBRSxDQUFDLHVEQUF1RCxDQUFDLENBQUM7SUFDbkUsT0FBTyxHQUFHLENBQUMsQ0FBQyxFQUFFLFFBQVEsRUFBRSxhQUFhLENBQUMsQ0FBQztBQUN6QyxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sS0FBSyxHQUFHLEVBQUUsQ0FBQyxFQUFDLE1BQU0sRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQge1RlbnNvcjREfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge2Fzc2VydH0gZnJvbSAnLi4vdXRpbCc7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3BhZH0gZnJvbSAnLi9wYWQnO1xuXG4vKipcbiAqIFBhZHMgYSBgdGYuVGVuc29yNERgIHdpdGggYSBnaXZlbiB2YWx1ZSBhbmQgcGFkZGluZ3MuIFNlZSBgcGFkYCBmb3IgZGV0YWlscy5cbiAqL1xuZnVuY3Rpb24gcGFkNGRfKFxuICAgIHg6IFRlbnNvcjREfFRlbnNvckxpa2UsXG4gICAgcGFkZGluZ3M6XG4gICAgICAgIFtcbiAgICAgICAgICBbbnVtYmVyLCBudW1iZXJdLCBbbnVtYmVyLCBudW1iZXJdLCBbbnVtYmVyLCBudW1iZXJdLCBbbnVtYmVyLCBudW1iZXJdXG4gICAgICAgIF0sXG4gICAgY29uc3RhbnRWYWx1ZSA9IDApOiBUZW5zb3I0RCB7XG4gIGFzc2VydChcbiAgICAgIHBhZGRpbmdzLmxlbmd0aCA9PT0gNCAmJiBwYWRkaW5nc1swXS5sZW5ndGggPT09IDIgJiZcbiAgICAgICAgICBwYWRkaW5nc1sxXS5sZW5ndGggPT09IDIgJiYgcGFkZGluZ3NbMl0ubGVuZ3RoID09PSAyICYmXG4gICAgICAgICAgcGFkZGluZ3NbM10ubGVuZ3RoID09PSAyLFxuICAgICAgKCkgPT4gJ0ludmFsaWQgbnVtYmVyIG9mIHBhZGRpbmdzLiBNdXN0IGJlIGxlbmd0aCBvZiAyIGVhY2guJyk7XG4gIHJldHVybiBwYWQoeCwgcGFkZGluZ3MsIGNvbnN0YW50VmFsdWUpO1xufVxuXG5leHBvcnQgY29uc3QgcGFkNGQgPSBvcCh7cGFkNGRffSk7XG4iXX0=