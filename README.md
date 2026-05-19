# nriet-rust-art

Py-ART-compatible radar toolkit with private Rust acceleration kernels.

The public Python API remains `import pyart`, and the distribution identity is
`arm_pyart`. Python keeps the mutable Radar/Grid object model. Rust kernels are
private implementation details exported through `pyart._rust`.

The initial demo namespace `nriet_rust_art` has been removed intentionally. This
repository now targets Py-ART compatibility, so downstream code should import
`pyart`.

## Compatibility Contract

- Public import: `import pyart`
- Distribution metadata: `arm_pyart`
- Private native module: `pyart._rust`
- Frozen Py-ART oracle: `F:\nriet-rust-art\pyart-main.zip`
- Operational RSTM parity data: `F:\nriet-rust-art\闽侯对比用数据`
- Default parity: exact shape, dtype, mask, fill value, metadata, warnings,
  exceptions, NaN positions, and numeric values

Any floating tolerance exception must be documented explicitly before it is
accepted.

Exact integer native slices currently include
`correct.bias_and_noise.calc_cloud_mask`'s private 4x4 count/threshold step;
it must match `signal.convolve2d(..., np.ones((4, 4)), mode="same")`
alignment exactly before applying `counts >= counts_threshold`. Shape, dtype,
data, and fallback exception behavior must remain exact.

Exact elementwise native slices currently include
`retrieve.qpe.est_rain_rate_zkdp` and `retrieve.qpe.est_rain_rate_za`'s
private threshold-blend assignment step. The Rust helper may only mutate dense,
unmasked, finite, writeable float64 main arrays and must preserve the original
`>`/`<` threshold selection and in-place `rain_main["data"]` semantics exactly.
Masked, nonfinite, noncontiguous, read-only, `thresh=None`, and non-boolean
`thresh_max` cases remain Python-owned.

`retrieve.qvp.project_to_vertical(..., interp_kind="none")` also has an exact
private dense projection helper. It only handles finite, unmasked, 1-D float64
C-contiguous inputs where `data_in` and `data_height` have equal length and the
grid has at least two points; masked/interpolating/exceptional cases remain on
the original Python path.

`retrieve.cfad.create_cfad` has an exact private normalization helper for
dense, finite, 2-D C-contiguous float64 frequency tables with at least one
altitude row and strictly positive row sums. It preserves the row-fraction
division, row-level threshold mask, masked-array dtype, shape, and fill value.
Zero-row, empty-row-sum, NaN, masked, noncontiguous, and non-float64 cases
remain Python-owned so NumPy warning, NaN, and exception behavior is unchanged.

`retrieve.simple_moment_calculations.compute_noisedBZ` uses exact private row
tiling helpers for 1-D float64 masked vectors. Data, mask, fill value, shape,
and signed-zero payloads must match `np.tile`; scalar, 2-D, negative-row, and
non-integer-row cases remain on the Python path.

`io._sigmetfile` has exact private helpers for native-endian dense SIGMET angle
arrays and ray-header parsing. Scalar angle calls, byte-swapped arrays,
zero-dimensional arrays, noncontiguous views, and malformed headers remain on
the Python path so view/dtype/exception behavior stays unchanged.

`io.cfradial._unpack_variable_gate_field_dic` has an exact private helper for
dense numeric 1-D CF/Radial variable-gate field arrays. Python still allocates
`np.ma.masked_all(shape, dtype=fdata.dtype)` and performs the final dictionary
mutation; Rust only copies valid source slices into the output payload and
clears the corresponding bool mask cells. Masked sources, nonnumeric/object
dtypes, noncontiguous arrays, metadata length truncation, negative indexes,
slice overflow, and other exceptional cases remain Python-owned so dtype, fill
value, mask, uninitialized masked-tail payload, and exception behavior remain
unchanged.

`aux_io.gamicfile._get_gamic_sweep_data` has exact private helpers for dense
GAMIC `UV8`, `UV16`, and `F` sweep arrays. The Rust path preserves the original
dynamic-range scale and offset formulas, raw-zero masks for unsigned integer
formats, NaN masks for float32 formats, float32 output dtype, shape, and default
masked-array fill value. Unknown formats, dtype mismatches, noncontiguous
arrays, object arrays, and nonfinite dynamic-range attributes remain
Python-owned so the original `assert`, `NotImplementedError`, warning, and
exception behavior is unchanged.

`aux_io.odim_h5._get_odim_h5_sweep_data` and
`aux_io.sinarame_h5._get_SINARAME_h5_sweep_data` share exact private helpers
for dense `uint8`/`uint16` ODIM-like sweep arrays whose gain/offset promotion
matches the original float64 NumPy path. Rust preserves raw-value masking order
(`nodata` before `undetect`), default gain/offset semantics, negative
gain/offset support, unscaled masked payload values, bool mask, output dtype,
shape, and `nodata` fill-value behavior. Float32 promotion cases, float raw
arrays, noncontiguous arrays, object/string arrays, NaN sentinels, nonfinite
gain/offset, and non-scalar attributes remain Python-owned so warning and
exception behavior is unchanged.

`aux_io.rainbow_wrl._get_data` has exact private helpers for dense
C-contiguous `uint8` and native `uint16` raw bins with complete `(nrays,
nbins)` payloads and `maxbin >= nbins`. Rust preserves the original
`float32` calibrated intermediate, final `float64` padded payload, raw-zero
mask, padded masked tail, fill value, and one-pass PhiDP/uPhiDP/uPhiDPu
`> 180` subtraction. Raw-zero payloads in the valid region keep the original
float32-rounded fill assignment, while padded tail cells keep the float64 fill
from `np.full`. Object arrays, scalar arrays, noncontiguous arrays, unsupported
dtypes, malformed dimensions, oversized native outputs, nonfinite scale inputs,
and exceptional reshape/broadcast cases remain Python-owned.

`aux_io.kazr_spectra._get_spectra` has an exact private row-gather helper for
dense C-contiguous float64 spectra arrays and float32/float64 locator vectors
whose finite indices are in bounds after Python `int(...)` truncation. Python
still owns the original `locator_mask` selection and in-place NaN to `-9999.0`
sentinel mutation before dispatch. Rust only copies selected spectra rows or
fills sentinel rows with NaNs into the original float64 output surface.
Out-of-bounds, nonfinite non-sentinel, object, noncontiguous, non-float64
spectra, oversized native outputs, and shape-mismatch cases remain Python-owned.
Object locator arrays retain the frozen oracle's pre-dispatch `np.isnan`
`TypeError` surface.

`io.output_to_geotiff._get_rgb_values` has an exact private helper for dense
2-D C-contiguous float64 data after Python resolves the Matplotlib colormap
into a 256-entry RGBA lookup table. Rust preserves the original value fraction,
index clamp, NumPy ties-to-even rounding, NaN RGB payloads, transparent alpha
behavior, no-NaN int64 RGB surface, and NaN-present float64 RGB surface.
Masked arrays, non-float64 data, non-2-D data, noncontiguous views, invalid
color/opacity parameters, and zero color range remain Python-owned.

`aux_io.edge_netcdf.read_edge_netcdf` has exact private mask builders for dense
C-contiguous native `uint8`, `uint16`, `int16`, `int32`, `float32`, and
`float64` field payloads with scalar numeric `MissingData` and `RangeFolded`
sentinels. Python still owns netCDF reads and Radar object construction; Rust
only returns the dense bool mask equal to the original existing mask OR sentinel
matches. Payload values, dtype, fill value, pre-existing masks, NaN sentinel
non-matches, and infinity equality are preserved. Byte-swapped, noncontiguous,
object/string sentinel, broadcast sentinel, complex, and other unsupported
cases remain Python-owned.

`io._sigmetfile.SigmetFile._get_ray` has an exact private helper for ray RLE
segments that are fully decodable within the current 3072-word SIGMET record.
It preserves native-endian `int16` record interpretation, missing-ray
`out[4] = -1`, zero-run corruption status `-1`, `rbuf_pos` state updates, and
in-place output mutation. Any ray that would require `_load_record` or a data
run beyond the output row remains Python-owned so cross-record file bookkeeping
and Python exception behavior stay unchanged.

`io._sigmetfile._data_types_from_mask` has an exact private helper for
non-boolean integer mask words in `0..=0xffffffff`. Negative Python integers,
oversized integers, booleans, and non-indexable objects remain Python-owned so
the original bit-shift behavior, including negative sign-bit semantics and
TypeError paths, is preserved.

`io.sigmet` has exact private helpers for the dense time-ordering predicates
used by reversal, roll, and reverse-roll detection, plus dense index-plan
helpers for applying `roll`, `reverse`, and stable full time sorting. Python
still owns `XHDR` vs. metadata time selection, all public data/metadata access,
the actual dict/array mutation, and unsupported inputs. Rust only handles 1-D
C-contiguous `int32` reference times and 1-D C-contiguous `int64`
`rays_per_sweep` arrays with nonnegative counts and at most 1,048,576 entries.
The helpers preserve the original no-advance behavior for 0/1-ray sweeps,
NumPy `int32` wrapping subtraction, first-minimum roll shifts, stable
mergesort-equivalent ties, and exact bool/index decisions for normal in-bounds
sweeps; negative, oversized, malformed, out-of-bounds, non-writable, and
non-dense mutation targets fall back to Python so original truncation and
exception behavior remains intact.

`io.uffile.UFFile._get_sweep_limits` has an exact private helper for dense
1-D C-contiguous `int32` ray sweep-number arrays. Rust preserves `np.unique`
sorted sweep-number order and returns the first and last positional ray index
for each sweep as `int32` arrays. Python still owns UF file parsing, object
state, `nsweeps` consistency, noncontiguous arrays, higher-rank arrays, object
and float edge cases, and all malformed/private-call exception behavior.

`io.uf_write.UFRayCreator._calc_ray_num_to_sweep_num` has an exact private
helper for real Py-ART `Radar` objects with dense 1-D C-contiguous `int32`
sweep start/end arrays and validated in-bounds inclusive ranges. Rust preserves
the zero-initialized `int32` ray map, zero-based enumerate sweep labels, gaps
remaining at sweep `0`, and later-sweep overwrite behavior for overlapping
ranges. Custom radar-like objects, non-dense metadata, negative or clipped
slice semantics, mismatched range vectors, and malformed inputs remain
Python-owned through the original `radar.iter_slice()` path.

`io.nexrad_level2.NEXRADLevel2File` has exact private helpers for dense
scan-message grouping and guarded message-index concatenation. The grouping
helper handles dense 1-D C-contiguous `int64` elevation-number arrays and
preserves the original `np.where(elev_nums == i + 1)[0]` list-of-arrays
surface, including missing elevation numbers as empty `int64` arrays, source
order within each scan, ignored nonpositive elevation numbers when positive
scans exist, and `nscans == len(scan_msgs)`. The concat helper handles normal
nonempty `range`/`list`/`tuple` scan selections over dense 1-D C-contiguous
`int64` `scan_msgs`, preserving scan order, duplicates, negative list-index
normalization, empty selected scans, and `np.concatenate` dtype/shape. Python
still owns radial-record selection, dictionary extraction, empty inputs,
higher-rank arrays, object, bool, float, malformed edge cases, empty scan
selection errors, custom iterables, ndarray scan selections, and unusual
`scan_msgs` mutation.

`io._sigmetfile.convert_sigmet_data` has an exact private dense helper for the
2-byte `like_dbt2` SIGMET family, including operational `DBZ2`. It handles
2-D C-contiguous native `int16` data with a valid nonnegative integer `nbins`
vector, preserves the `uint16` view conversion `(N - 32768) / 100`, masks raw
zero gates and uncollected tail gates, and returns the same `float32` data,
bool mask, `fill_value=-9999.0`, and `shrink=False` masked-array surface.
Other data families, masked arrays, byte-swapped or noncontiguous arrays,
wrong ranks, malformed `nbins`, and negative `nbins` remain Python-owned.

`io._sigmetfile.convert_sigmet_data` also has an exact private dense helper for
the 1-byte `like_dbt` SIGMET family. It preserves the native byte-view
semantics of `data.view('(2,) uint8').reshape(nrays, -1)[:, :nbin]`, applies
`(N - 64) / 2`, masks raw byte zero and uncollected tail gates, and returns the
same `float32` data, bool mask, `fill_value=-9999.0`, and `shrink=False`
surface. Zero-ray inputs, which the Python oracle rejects during reshape,
remain Python-owned, as do other unsupported dense cases.

`io._sigmetfile.convert_sigmet_data` also has exact private dense helpers for
the one-byte `like_sqi`, `VEL`, `VELC`, `WIDTH`, `ZDR`, `KDP`, `PHIDP`, and
`HCLASS` branches. They preserve the same native byte-view surface as
`like_dbt`, including zero-ray fallback to the Python reshape error. The
`like_sqi` helper only dispatches when the complete output byte view has no raw
zero values; raw-zero cases remain Python-owned so NumPy's pre-mask
`sqrt` warning and NaN payload are preserved. Unknown one-byte branches remain
Python-owned.

`io._sigmetfile.convert_sigmet_data` also has exact private dense helpers for
the 2-byte `like_sqi2`, `WIDTH2`, `PHIDP2`, and `HCLASS2` SIGMET branches.
They preserve each branch's `uint16` view formula and tail masking. `like_sqi2`,
`WIDTH2`, and `PHIDP2` mask raw zero gates; `HCLASS2` does not mask raw zero
and only applies the uncollected-tail mask. Unsupported dense cases remain
Python-owned under the same guards as the `like_dbt2` helper.

`io.mdv_common._decode_rle8` has an exact private MDV RLE8 decoder for
well-formed `bytes` payloads whose decoded length exactly matches
`decompr_size` and is no larger than the 512 MiB native safety cap.
All-literal payloads may use that full cap; payloads containing run packets are
native-owned only while the decoded pointer stays within `0..=255`, because the
Python oracle's `np.uint8` run count can make later write offsets wrap.
Bytearray/memoryview inputs, non-integer or out-of-range keys,
negative/non-integer sizes, oversized outputs, truncated runs, overflow,
underfilled outputs, and large run/mixed payloads remain Python-owned to
preserve the original exception, warning, and uninitialized-tail surface.

`io.nexrad_level3` has an exact private AF1F radial RLE decoder for
well-formed `bytes` rows whose high-nibble run lengths sum exactly to `nbins`
and remain under the 512 MiB native safety cap. Bytearray/memoryview rows,
non-integer or negative bin counts and decoded length mismatch remain
Python-owned so the existing `radial[:] = np.repeat(...)` broadcast exception
surface is unchanged. Oversized direct helper outputs are rejected before
falling back to `np.repeat` so the private helper cannot allocate beyond the
native safety cap.

`io.nexrad_level3.NEXRADLevel3File._get_data_8_or_16_levels` has an exact
private threshold-table helper for 2-D C-contiguous `uint8` raw data, complete
32-byte threshold tables, and raw indices in `0..15`. Short threshold tables,
non-`uint8` or non-2-D raw arrays, noncontiguous views, and out-of-range raw
indices remain Python-owned so `np.choose` dtype and exception behavior stays
unchanged.

`io.nexrad_level3.NEXRADLevel3File._get_data_msg_134` has an exact private
message-134 scaling helper for 2-D C-contiguous `uint8` raw data and at least
10 threshold bytes when the decoded linear/log scales are finite and nonzero.
Short threshold tables, zero-scale warning paths, non-`uint8` or non-2-D raw
arrays, and noncontiguous views remain Python-owned to preserve NumPy warning
and exception behavior.

`io.nexrad_level3.NEXRADLevel3File.get_data` has exact private
message-32 and message-94/99/153/154/155/182/186 scaling helpers for 2-D
C-contiguous `uint8` or `uint16` raw data and threshold tables with at least
four bytes. The subtract-two helper preserves the original unsigned
wraparound payload for masked raw gates `0` and `1`. Short threshold tables,
non-unsigned integer raw arrays, non-2-D arrays, and noncontiguous views remain
Python-owned.

`io.nexrad_level3.NEXRADLevel3File.get_data` also has exact private raw-data
helpers for message-165/177 classification masks and message-34 copies on 2-D
C-contiguous `uint8` or `uint16` raw data. Message 165/177 preserves
`np.ma.masked_equal(..., 0)` fill value `0.0`; message 34 preserves the copied,
unmasked raw payload. Other dtypes, ranks, and noncontiguous views remain
Python-owned.

`io.nexrad_level3.NEXRADLevel3File.get_data` also has an exact private
message-135 helper for 2-D C-contiguous `uint8` raw data. It preserves the
original unsigned wraparound sequence `raw_data - 2` followed by subtracting
`np.uint8(128)` for raw gates `>= 128`, then returns the same float32 data and
`raw_data <= 1` mask after the public `astype("float32")` step. Non-`uint8`,
non-2-D, and noncontiguous cases remain Python-owned.

`io.nexrad_level3.NEXRADLevel3File.get_data` also has an exact private
message-138 linear-scaling helper for 2-D C-contiguous `uint8` raw data and
threshold tables with at least four bytes. Short threshold tables,
non-`uint8` or non-2-D raw arrays, and noncontiguous views remain Python-owned
to preserve the original NumPy dtype and exception behavior.

`io.nexrad_cdm._get_moment_data` has exact private helpers for dense
Common Data Model NEXRAD moment slices whose scale/offset promotion matches the
original float64 NumPy path. Python still owns `set_auto_maskandscale(False)`,
raw scan slicing, and `_Unsigned` int8/int16 views before dispatch. Rust
handles dense `uint8`, `uint16`, `int8`, `int16`, `float32`, and `float64`
arrays while preserving the `raw <= 1` mask, unscaled masked payload values,
NaN comparison behavior, scale/offset formula, output dtype, shape, and default
masked-array fill value. Dispatch is intentionally limited to inputs where
`np.result_type(raw.dtype, scale, add_offset)` is exactly `float64`.
Float32-promotion attributes, noncontiguous arrays, nonfinite scale/offset, and
non-scalar attributes remain Python-owned.

`io.chl.ChlFile._extract_fields` has an exact private helper for integer CHL
field payloads inside multi-field structured ray records. Rust decodes native
endian `uint8`, `uint16`, and `uint64` fields, preserves the raw-zero mask,
leaves masked payload data at `0.0`, applies the original
`(data * dat_factor + dat_bias) / fld_factor` formula to unmasked gates, and
keeps the pre-`read_chl` fill value as the original integer zero dtype. Python
still owns float32 fields so `np.ma.masked_values(..., 0)` tolerance is
unchanged. Single-field records, incomplete payloads, unsafe scale factors,
unknown format codes, and missing Rust remain Python-owned.
Single-field records retain the frozen Py-ART oracle's original exception
surface when `_extract_fields` indexes `dtype.names`.

`correct.phase_proc.unwrap_masked` has an exact private helper for the
compressed valid sequence in one-dimensional float64 degree inputs after Python
has applied `np.ma.masked_invalid(...).astype(float)`. Python still owns the
MaskedArray shell, original fill value restoration, invalid masking, early
returns for short or single-valid inputs, `centered=True` NumPy half-even
rounding, the frozen oracle's accepted-but-unused `copy` flag behavior, and the
final masked/unmasked return surface. Rust only unwraps the finite unmasked
sequence using the original strict `diff > 180` and
`diff < -180` period decisions, preserving gaps, exact `+/-180` thresholds, and
valid-point order. Higher-rank, short, non-float64, noncontiguous, nonfinite
unmasked, and malformed inputs remain Python-owned.

`correct.phase_proc.smooth_and_trim_scan` has an exact private helper for the
2-D float64 scan-direction smoothing core that matches
`scipy.ndimage.convolve1d(..., axis=1, mode="reflect", origin=0)` after Python
builds and normalizes the window. Python still owns window construction,
`window_len < 3` early return, invalid-window exceptions, and SciPy fallback
for masked, non-float64, noncontiguous, nonfinite, short-width, and malformed
inputs. Rust uses SciPy-compatible mirror reflect indexing per row; it does not
reuse the 1-D `smooth_and_trim` custom reflect padding. The `sg_smooth` fast
path is enabled only for `window_len == 5`.

`correct.attenuation.get_mask_fzl` has an exact private helper for the
temperature/height-over-iso0 `gate_excluded` scan that derives one `int32`
`end_gate_arr` value per ray. Rust handles only dense 2-D C-contiguous bool
masks whose shape exactly matches `(radar.nrays, radar.ngates)`, preserving the
first-excluded-gate rule, first-gate clamp to `0`, no-excluded `ngates - 1`,
zero-gate `-1`, and row order. Non-bool, noncontiguous, masked, shape-drift,
and malformed inputs remain Python-owned so the frozen oracle's indexing and
exception behavior is unchanged.

`core.transforms` has exact private `float32` helpers for
`_interpolate_axes_edges` and `_interpolate_range_edges` on 1-D C-contiguous
arrays with at least two centers. Range edges apply the original negative clamp
after all edge values are computed, leaving NaNs unchanged. Other dtypes,
short arrays, multidimensional arrays, and noncontiguous views remain
Python-owned.

`map.GateMapper.mapped_radar` has an exact private helper for the inner
float64 field assignment loop. Python still owns Radar copying, field-list
normalization, field metadata, target masked-array construction, and source
mask merging with `gatefilter_src.gate_excluded`. Rust only handles finite,
dense, C-contiguous `float64` index maps with shape
`(src_nrays, src_ngates, 2)`, dense source data/mask arrays, and writable
float64 destination data plus bool masks. It preserves row-major source-gate
order, later duplicate overwrites, destination ray `0` skip behavior, and
masked-source assignments that set the destination mask while leaving the
payload value unchanged. Non-float64 fields, noncontiguous arrays, NaNs,
negative wrapping columns, positive out-of-bounds indexes, and malformed
objects remain Python-owned so oracle indexing and exception behavior stays
unchanged.

`retrieve._echo_class_wt.label_classes` has an exact private helper for the
dense float64 pre-cast threshold-classification surface. Python still owns
masked-array data extraction, broadcast/object/non-float fallback, and the
final `.astype(np.int32)` so NaN-to-int warning and `np.seterr` behavior remain
NumPy-owned. Rust preserves the frozen ordered `np.where` semantics, including
the overwritten first convective-core pass, `>=`/`<` threshold boundaries,
underlying-data behavior for masked inputs, and `1.0`/`2.0`/`3.0`/`NaN`
precursor values. Shape-broadcasting, noncontiguous arrays, non-float64 arrays,
and nonnumeric thresholds remain Python-owned.

`util.simulated_vel.simulated_vel_from_profile` has an exact private dense
radial-velocity helper after Python completes the original SciPy wind-profile
interpolation and `np.ma.masked_invalid` handling. Python still owns Radar and
profile objects, metadata, interpolation, invalid-wind masking, and all masked
or exceptional cases. Rust only evaluates the dense float64 row-wise formula
using Python-computed sine/cosine vectors, preserving output shape, dtype,
full-false mask surface, fill value `1e20`, and numeric values exactly for
finite C-contiguous two-dimensional gate wind arrays.

`retrieve.advection.grid_displacement_pc` has an exact private peak-extraction
helper after Python completes FFT phase-correlation and `fftshift`. Python
still owns Grid objects, masked fill handling, FFTs, distance/velocity
conversion, and NaN/empty/non-2-D exception paths. Rust only scans dense
finite 2-D C-contiguous float64 correlation images, preserving NumPy
row-major first-maximum tie-breaking, floor center subtraction, and the
public `np.int64` pixel displacement surface.

`util.radar_utils.image_mute_radar` has an exact private dense bool
mask-construction helper for plain C-contiguous float64 field pairs with
identical shape. Python still owns Radar mutation, field metadata copies,
`np.ma.masked_where`, `masked_invalid`, existing masked-array semantics,
shape/broadcast exceptions, and all unsupported dtypes. Rust only evaluates
the inclusive `data_mute_by <= mute_threshold` and optional
`data_to_mute >= field_threshold` comparisons under the same NaN/inf
comparison rules as NumPy.

`util.columnsect.get_sweep_rays` has an exact private dense index-scan helper
for 1-D C-contiguous finite float64 sweep-azimuth arrays with at least two rays
and at most 1,048,576 rays. Python still computes the original
`np.round(..., 3)` azimuth resolution, owns list conversion, and keeps large
inputs, list/object, masked, non-float64, noncontiguous, nonfinite,
short-array, and exceptional paths on the original NumPy implementation. Rust
only returns the natural-order indices satisfying the strict `< 0.5`
centerline and strict `< resolution * azimuth_spread` spread comparisons.

`util.columnsect.get_column_rays` has exact private dense helpers for PPI
nearest-ray selection and RHI ray filtering on 1-D C-contiguous float64 azimuth
arrays with at most 1,048,576 rays. Python still owns the public azimuth
type/range checks, Radar object access, sweep iteration, empty-result
`ValueError`, and unsupported array/metadata paths. The PPI helper preserves
`np.argmin(np.abs(sweep_azi - azimuth))` first-tie and first-NaN behavior; the
RHI helper preserves the original sweep `range(nstart, nstop)` end-exclusion
and strict `< 1` azimuth comparison.

`util.xsect.cross_section_ppi` and `util.xsect.cross_section_rhi` share an exact
private nearest-angle helper for 1-D C-contiguous float64 sweep-angle arrays
with at most 1,048,576 rays. Python still owns the existing PPI sorted-target
traversal, PPI unique fixed-angle construction, RHI input-order traversal,
tolerance warnings and errors, Radar construction, metadata copying, and
unsupported inputs. Rust only returns `np.argmin(np.abs(values - target))` and
`np.min(np.abs(values - target))`, preserving first-tie and first-NaN behavior
for the native path.

`retrieve.cappi.create_cappi` has an exact private nearest-height index helper
for dense 3-D C-contiguous float64 height volumes. Python still owns Radar
objects, sweep/ray selection, interpolation, height tolerance derivation,
selected data gathering, masks, valid range metadata, and output assembly.
Rust only returns the `np.argmin(np.abs(z_3d - height), axis=0)` index surface
and matching selected gate heights, preserving int64 indexes, float64 selected
heights, row-major first-tie behavior, and NumPy's first-NaN argmin behavior.
Masked, non-float64, noncontiguous, empty-sweep, and exceptional cases remain
Python-owned.

`map._gate_to_grid_map.GateToGridMapper` has exact private helpers for dense
gate-to-grid bounds, single-gate accumulation, and ROI-grid filling. The ROI
helper handles writeable C-contiguous `float32` `(nz, ny, nx)` outputs for the
built-in `ConstantRoI`, `DistRoI`, and `DistBeamRoI` classes, preserving the
original `ix`/`iy`/`iz` coordinate order, `roi_array[iz, iy, ix]` write
surface, minimum-over-offset semantics, empty-offset sentinel, and
`DistBeamRoI`'s float32 `h_factor` intermediate rounding before assignment.
Custom ROI classes, non-float32 outputs, malformed shapes, noncontiguous or
read-only arrays, nonfinite parameters, and unsupported object state remain on
the original Python loop.

## Floating Tolerance Exceptions

- `retrieve.simple_moment_calculations.compute_cdr`: the dense Rust CDR kernel
  may differ from NumPy `pow`/`sqrt`/`log10` by at most `1.0e-13` absolute on
  unmasked finite values. Shape, dtype, mask, fill value, metadata, warnings,
  and exceptions remain exact.
- `retrieve.simple_moment_calculations.compute_l`: the dense Rust logarithmic
  cross-correlation ratio kernel may differ from NumPy `ma.log10` by at most
  `1.0e-14` absolute on guarded finite values. The original in-place
  `rhohv >= 1.0` clamp side effect, shape, dtype, mask, fill value, metadata,
  warnings, and exceptions remain exact.
- `correct.bias_and_noise.cloud_threshold`: the dense one-dimensional Rust
  kernel may differ from NumPy `power`/`sqrt`/`log10` by at most `1.0e-12`
  absolute for finite in-range rows. Scalar type, warnings, and exceptions stay
  on the Python fallback path unless the strict dense guard accepts the input.
- `correct.bias_and_noise.correct_noise_rhohv`: the dense Rust elementwise
  RhoHV noise-correction kernel may differ from NumPy `ma.power`/`ma.sqrt` by
  at most `1.0e-12` absolute for finite guarded inputs with all dB-domain
  operands and noise-difference terms within `[-300, 300]`. Masked, nonfinite,
  overflow-risk, shape-mismatch, and metadata/error paths remain Python-owned.
- `util.sigmath.texture_along_ray`: the dense Rust rolling population-standard
  deviation kernel may differ from NumPy `ma.std` by at most `1.0e-14`
  absolute for finite unmasked 2-D float64 fields. Masked/nonfinite and
  exceptional window cases remain on the Python fallback path.
- `retrieve.qpe.est_rain_rate_zpoly`: the dense Rust polynomial/power kernel
  may differ from NumPy `ma.power` by at most `1.0e-12` absolute for finite
  unmasked float64 reflectivity arrays in the guarded range. Masked/nonfinite
  and metadata/error paths remain Python-owned.
- `retrieve.qpe.est_rain_rate_z`: the dense Rust power-law rain-rate kernel may
  differ from NumPy's nested `ma.power` expression by at most `1.0e-12`
  absolute for finite guarded inputs. Overflow, masked, nonfinite, and unusual
  coefficient paths remain on the Python fallback path.
- `retrieve.qpe.est_rain_rate_kdp`: the dense Rust KDP power-law rain-rate
  kernel may differ from NumPy `ma.power` by at most `1.0e-12` absolute for
  finite guarded inputs after the public Python path performs its original
  in-place negative-KDP clamp. Overflow, masked, nonfinite, and unusual
  coefficient paths remain on the Python fallback path.
- `retrieve.qpe.est_rain_rate_a`: the dense Rust attenuation power-law
  rain-rate kernel may differ from NumPy `ma.power` by at most `1.0e-12`
  absolute for finite guarded nonnegative inputs. Negative, overflow, masked,
  nonfinite, and unusual coefficient paths remain on the Python fallback path.
- `retrieve.spectra_calculations._get_mean_velocity`: the dense Rust spectra
  moment kernel may differ from the NumPy trapezoid/nansum path by at most
  `1.0e-12` absolute for guarded 2-D/3-D spectra. One-dimensional, masked,
  nonfinite, and shape-broadcast cases remain on the Python fallback path.
- `retrieve.spectra_calculations._get_spectral_width`: the dense Rust spectra
  moment kernel may differ from the NumPy trapezoid/nansum/sqrt path by at
  most `1.0e-12` absolute for guarded 2-D/3-D spectra. One-dimensional,
  masked, nonfinite, and shape-broadcast cases remain Python-owned, including
  the oracle exception/warning surface.
- `retrieve.spectra_calculations._get_skewness`: the dense Rust spectra
  shape-moment kernel may differ from the NumPy trapezoid/nansum/divide path
  by at most `1.0e-12` absolute for guarded 2-D/3-D spectra. Zero-width,
  one-dimensional, masked, nonfinite, and shape-broadcast cases remain on the
  Python fallback path, preserving the oracle exception/warning surface.
- `retrieve.spectra_calculations._get_kurtosis`: the dense Rust spectra
  shape-moment kernel may differ from the NumPy trapezoid/nansum/divide path
  by at most `1.0e-12` absolute for guarded 2-D/3-D spectra. Zero-width,
  one-dimensional, masked, nonfinite, and shape-broadcast cases remain on the
  Python fallback path, preserving the oracle exception/warning surface.
- `util.circular_stats.compute_directional_stats`: the dense Rust directional
  mean helper may differ from NumPy `ma.mean` by at most `1.0e-14` absolute for
  unmasked finite 2-D float64 C-contiguous arrays along axis 0 or 1. Masked,
  nonfinite, noncontiguous, empty, non-2-D, negative-axis, median, and other
  unsupported cases remain on the Python fallback path.
- `retrieve.vad._vad_calculation_m`: the dense Rust Michelson VAD helper may
  differ from the NumPy masked-array least-squares path by at most `1.0e-12`
  absolute for finite unmasked 2-D float64 velocity fields with a positive even
  number of rays and non-singular azimuth design terms. Masked, NaN, odd-ray,
  noncontiguous, singular, and exceptional inputs remain on the Python fallback
  path.

## Development

The repository is expected at:

```powershell
F:\nriet-rust-art\repo
```

The frozen oracle zip and operational data live outside the repo and must not be
committed.

Build a wheel and install it into the active Python environment:

```powershell
python -m maturin build --release
python -m pip install --force-reinstall --no-deps target\wheels\arm_pyart-0.1.0-cp312-cp312-win_amd64.whl
```

Run the source-tree validation suite before native install:

```powershell
$env:PYART_QUIET='1'
$env:PYART_ALLOW_MISSING_RUST='1'
python -m pytest tests -q
```

Run installed-wheel parity, which is the native extension acceptance gate:

```powershell
$env:PYART_QUIET='1'
$env:PYART_TEST_INSTALLED='1'
$env:RSTM_DATA_ROOT='F:\nriet-rust-art\闽侯对比用数据'
python -m pytest tests -q
```

Run operational RSTM validation when the data is available:

```powershell
$env:RSTM_DATA_ROOT='F:\nriet-rust-art\闽侯对比用数据'
python -m pytest tests\rstm\test_minhou_operational.py -q
```

## Current Native Slices

- `pyart._rust.rust_backend_ready`
- `pyart._rust.sum_f64`
- gzip byte helpers for RSTM freezing
- RSTM header preview helpers
- `_kdp_proc.lowpass_maesaka_term`
- `_kdp_proc.lowpass_maesaka_jac`
- `_kdp_proc.forward_reverse_phidp`
- `_kdp_proc` uniform range-resolution helper
- `core.transforms` antenna/cartesian, AEQD coordinate, and float32 edge-interpolation kernels
- `correct` unwrap 1D/2D/3D, fast edge finder, first-mask, cloud-threshold, cloud-mask 4x4 count, RhoHV noise correction, and region-cost kernels
- `correct._common_dealias` dense limit-setting helper
- `correct.despeckle` 360-degree sweep check kernel
- `correct.attenuation` PhiDP preparation, end-gate mask scan, and default-parameter table kernels
- `correct.phase_proc` masked degree unwrap, 1-D/2-D dense smoothing, freezing-level, and system-phase kernels
- `filters.GateFilter` mask merge plus threshold/interval/finite compare-merge kernels
- `map.GateMapper` dense float64 field-assignment kernel
- `retrieve.echo_class` no-entropy scan classifier, ATWT, wavelet threshold labels, radial mask, feature classification, scalar core scheme, frequency-band, standardization, and feature-radius kernels
- `retrieve.cfad` dense frequency-table normalization helper
- `retrieve.qpe` default coefficient table plus Z-poly, Z power-law, KDP power-law, attenuation power-law, and Z/KDP or Z/A threshold-blend kernels
- `retrieve.qvp` finite range/angle index, nearest-gate, neighbour-gate, and vertical projection helper kernels
- `retrieve.simple_moment_calculations` SNR, dense and masked noise tiling, logarithmic cross-correlation ratio, circular-depolarization ratio, and linear-depolarization clamp helpers
- `retrieve.spectra_calculations` dealiased spectra, peak-limit, reflectivity, mean-velocity, spectral-width, skewness, and kurtosis kernels
- `retrieve.srv` storm-relative velocity row-update kernel
- `retrieve.vad` `_Average1D`, inverse-distance, interval-mean, dense Browning VAD, and dense Michelson VAD kernels
- `retrieve.advection` dense phase-correlation peak extraction helper
- `retrieve.cappi` dense nearest-height index helper
- `io.nexrad_interpolate` scan interpolation kernels
- `io.nexrad_level3` NEXRAD 16-bit float decode helper
- `io.nexrad_level3` AF1F radial RLE decode helper
- `io.nexrad_level3` 8/16-level threshold-table helper
- `io.nexrad_level3` message-134 scaling helper
- `io.nexrad_level3` message-32 and message-94/99/153/154/155/182/186 scaling helpers
- `io.nexrad_level3` message-165/177 mask-zero and message-34 copy helpers
- `io.nexrad_level3` message-135 wraparound helper
- `io.nexrad_level3` message-138 linear-scaling helper
- `io.nexrad_cdm` dense numeric moment scale/mask helper
- `io.chl` multi-field integer payload extraction helper
- `aux_io.gamicfile` GAMIC UV8/UV16/F sweep decode helpers
- `aux_io.rainbow_wrl` RAINBOW uint8/uint16 raw-bin calibration helper
- `aux_io.kazr_spectra` dense float64 spectra row-gather helper
- `aux_io.edge_netcdf` dense numeric MissingData/RangeFolded mask helpers
- `aux_io.odim_h5` and `aux_io.sinarame_h5` ODIM-like uint8/uint16 sweep decode helpers
- `io.output_to_geotiff` dense float64 RGB lookup helper
- `io.cfradial` dense CF/Radial variable-gate unpack helper
- `io.mdv_common` MDV RLE8 decode helper
- `io._sigmetfile` uncollected-gate masking kernel
- `io._sigmetfile` SIGMET current-record ray RLE decode helper
- `io._sigmetfile` SIGMET angle and ray-header parsing helpers
- `io._sigmetfile` data-type mask expansion helper
- `io._sigmetfile` SIGMET like-dbt2 dense data conversion helper
- `io._sigmetfile` SIGMET like-dbt dense data conversion helper
- `io._sigmetfile` SIGMET one-byte simple dense data conversion helpers
- `io._sigmetfile` SIGMET simple 2-byte dense data conversion helpers
- `io.sigmet` dense time-ordering predicate and index-plan helpers
- `io.uffile` dense UF sweep-limit helper
- `io.uf_write` dense UF ray-to-sweep map helper
- `io.nexrad_level2` dense scan-message grouping and message-index concat helpers
- `map._load_nn_field_data` and gate-to-grid map/ROI helper kernels
- `util.sigmath.angular_texture_2d` and `texture_along_ray`
- `util.simulated_vel` dense radial-velocity helper
- `util.radar_utils.image_mute_radar` dense mask-construction helper
- `util.columnsect` dense sweep-ray and column-ray index helpers
- `util.xsect` dense cross-section nearest-angle helper
- `util.hildebrand_sekhon.estimate_noise_hs74`
- `util.circular_stats` circular and interval statistic kernels

More Py-ART algorithms are migrated slice by slice behind the same Python module
names, with parity tests added before replacing behavior.

## Verification Results (2026-05-19)

Recorded on branch `rust-kernel-full-rewrite` after bootstrap baseline and
`smooth_and_trim_scan` native slice. Re-run these gates after further changes.

| Gate | Command | Result |
|------|---------|--------|
| Rust format | `cargo fmt --check` | see CI log below |
| Rust tests | `cargo test -q` | see CI log below |
| Source pytest | `python -m pytest tests -q` | see CI log below |
| Wheel | `maturin build --release --out dist` | see CI log below |
| Installed pytest | `PYART_TEST_INSTALLED=1 python -m pytest tests -q` | see CI log below |
| RSTM | `RSTM_DATA_ROOT=...\闽侯对比用数据` + installed pytest | see CI log below |
