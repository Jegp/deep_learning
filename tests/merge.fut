import "../lib/deep_learning"
import "../lib/util"

module dl = deep_learning f64
module util = utility f64

let merge = dl.layers.merge ([3, 3],6) dl.nn.identity 1

let apply_grad_gd (alpha:f64)
                  (batch_size:i32)
                  ((w,b):([][]f64, []f64))
                  ((wg,bg):([][]f64,[]f64)) =

  let wg_mean   = map (map f64.((/i32 batch_size))) wg
  let bg_mean   = map (f64.((/i32 batch_size))) bg

  let wg_scaled = util.scale_matrix wg_mean alpha
  let bg_scaled = util.scale_v bg_mean alpha

  let w'        = util.sub_matrix w wg_scaled
  let b'        = util.sub_v b bg_scaled

  in (w', b')

let updater = (apply_grad_gd 0.1 1)


-- ==
-- entry: merge_fwd
-- input {[[[1.0, 2.0, 3.0, 4.0],
--          [2.0, 3.0, 4.0, 5.0],
--          [3.0, 4.0, 5.0, 6.0]],
--	   [[1.0, 2.0, 3.0, 4.0],
--          [2.0, 3.0, 4.0, 5.0],
--          [3.0, 4.0, 5.0, 6.0]]]
-- 
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[[ 31.0,  72.0, 113.0],
--          [ 41.0,  98.0, 155.0],
--          [ 51.0, 124.0, 197.0],
--	    [ 31.0,  72.0, 113.0],
--          [ 41.0,  98.0, 155.0],
--          [ 51.0, 124.0, 197.0]]}

entry merge_fwd input w b =
  let weights = replicate 2 (w, b)
  let (_, output) = merge.forward false weights input
  in output

-- ==
-- entry: merge_bwd_err
-- input {[[[1.0, 2.0, 3.0, 4.0],
--          [2.0, 3.0, 4.0, 5.0],
--          [3.0, 4.0, 5.0, 6.0]],
--         [[1.0, 2.0, 3.0, 4.0],
--          [2.0, 3.0, 4.0, 5.0],
--	    [3.0, 4.0, 5.0, 6.0]]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[[[1408.0, 1624.0, 1840.0, 2056.0],
--           [1926.0, 2220.0, 2514.0, 2808.0],
--           [2444.0, 2816.0, 3188.0, 3560.0]],
--          [[1408.0, 1624.0, 1840.0, 2056.0],
--           [1926.0, 2220.0, 2514.0, 2808.0],
--           [2444.0, 2816.0, 3188.0, 3560.0]]]}

entry merge_bwd_err input w b =
  let weights = replicate 2 (w, b)
  let (cache, output) = merge.forward true weights input
  let (err, _) = merge.backward false updater weights cache output
  in err

-- ==
-- entry: merge_bwd_dW
-- input {[[[1.0, 2.0, 3.0, 4.0],
--          [2.0, 3.0, 4.0, 5.0],
--          [3.0, 4.0, 5.0, 6.0]],
--         [[1.0, 2.0, 3.0, 4.0],
--          [2.0, 3.0, 4.0, 5.0],
--	    [3.0, 4.0, 5.0, 6.0]]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[[[-25.60,  -36.90,  -48.20,  -59.50],
--           [-59.00,  -87.40, -115.80, -144.20],
--           [-92.40, -137.90, -183.40, -228.90]],
--          [[-25.60,  -36.90,  -48.20,  -59.50],
--           [-59.00,  -87.40, -115.80, -144.20],
--           [-92.40, -137.90, -183.40, -228.90]]]}


entry merge_bwd_dW input w b =
  let weights = replicate 2 (w, b)
  let (cache, output) = merge.forward true weights input
  let (_, weights_out) = merge.backward false updater weights cache output
  let (w', _) = unzip weights_out
  in w'

-- ==
-- entry: merge_bwd_dB
-- input {[[[1.0, 2.0, 3.0, 4.0],
--          [2.0, 3.0, 4.0, 5.0],
--          [3.0, 4.0, 5.0, 6.0]],
--         [[1.0, 2.0, 3.0, 4.0],
--          [2.0, 3.0, 4.0, 5.0],
--	    [3.0, 4.0, 5.0, 6.0]]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[[-11.30, -27.40, -43.50], [-11.30, -27.40, -43.50]]}

entry merge_bwd_dB input w b =
  let weights = replicate 2 (w, b)
  let (cache, output) = merge.forward true weights input
  let (_, weights_out) = merge.backward false updater weights cache output
  let (_, b') = unzip weights_out
  in b'

