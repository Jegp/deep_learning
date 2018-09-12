import "../lib/deep_learning"
import "../lib/util"

module dl = deep_learning f64
module util = utility f64

let replicate = dl.layers.replicate 4 dl.nn.identity 1

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
-- entry: replicate_fwd
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[[ 31.0,  72.0, 113.0],
--           [ 41.0,  98.0, 155.0],
--           [ 51.0, 124.0, 197.0]]
--
--          [[ 31.0,  72.0, 113.0],
--           [ 41.0,  98.0, 155.0],
--           [ 51.0, 124.0, 197.0]]}

entry replicate_fwd input w b =
  let ws = ((w, b), (w, b))
  let (_, output) = replicate.forward false ws input
  in output


-- ==
-- entry: replicate_backward_w
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[[-25.60,  -36.90,  -48.20,  -59.50],
--          [-59.00,  -87.40, -115.80, -144.20],
--          [-92.40, -137.90, -183.40, -228.90]]}


entry replicate_backward_w input w b =
  let ws = ((w, b), (w, b))
  let (cache, output) = replicate.forward true ws input
  let (_, ((w',_), (_, _))) = replicate.backward false updater ws cache output
  in w'

-- ==
-- entry: replicate_backward_b
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[-11.30, -27.40, -43.50]}

entry replicate_backward_b input w b =
  let ws = ((w, b), (w, b))
  let (cache, output) = replicate.forward true ws input
  let (_, ((_,b'), (_, _))) = replicate.backward false updater ws cache output
  in b'
