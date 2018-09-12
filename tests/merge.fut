import "../lib/deep_learning"
import "../lib/util"

module dl = deep_learning f64
module util = utility f64

let merge = dl.layers.merge (3, 1) dl.nn.identity 1

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
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0, 2.0, 3.0, 4.0]]}
--
-- output {[[ 1.0, 2.0, 3.0, 4.0],
--          [ 2.0, 3.0, 4.0, 5.0],
--          [ 3.0, 4.0, 5.0, 6.0],
--	    [ 1.0, 2.0, 3.0, 4.0]]}

entry merge_fwd i1 i2 =
  let (_, output) = merge.forward false () (i1, i2)
  in output

-- ==
-- entry: merge_bwd_err
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0, 2.0, 3.0, 4.0]]}
--
-- output {[[1.0, 2.0, 3.0, 4.0],
--          [2.0, 3.0, 4.0, 5.0],
--          [3.0, 4.0, 5.0, 6.0],
--          [1.0, 2.0, 3.0, 4.0]]}

entry merge_bwd_err i1 i2 =
  let (cache, output) = merge.forward true () (i1, i2)
  let ((e1, e2), _) = merge.backward false updater () () output
  in e1 ++ e2

-- ==
-- entry: merge_bwd_dW
-- input {[[1.0, 2.0, 3.0, 4.0],
--         [2.0, 3.0, 4.0, 5.0],
--         [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0]]}
--
-- output {}

entry merge_bwd_dW i1 i2 =
  let (cache, output) = merge.forward true () (i1, i2)
  let (_, w') = merge.backward false updater () () output
  in w'

