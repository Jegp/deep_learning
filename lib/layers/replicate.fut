import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "/futlib/linalg"

-- | Split input into several layers
module replicate (R:real) : layer_type with t = R.t
                                   with input_params = (i32, []i32)
                                   with activations  = activation_func ([]R.t)
                                   with input        = arr2d R.t
                                   with weights      = std_weights R.t
                                   with output       = [](arr2d R.t)
                                   with cache        = [](arr2d R.t, arr2d R.t)
                                   with error_in     = []arr2d R.t
                                   with error_out    = arr2d R.t = {

  type t            = R.t
  type input        = arr2d t
  type weights      = (std_weights R.t)
  type output       = []arr2d t
  type cache	    = [](arr2d t, arr2d t)
  type error_in     = []arr2d t
  type error_out    = arr2d t
  type b_output     = (error_out, std_weights R.t)

  type input_params = (i32, []i32)
  type activations  = activation_func ([]t)

  type replicate_nn = NN input weights output
                      cache error_in error_out 
		      ((std_weights t) -> (std_weights t) -> (std_weights t))

  module lalg   = linalg R
  module util   = utility R
  module w_init = weight_initializer R

  let empty_cache : (arr2d t, arr2d t) = ([[]],[[]])
  let empty_error : error_out = [[]]

  -- Forward propagation
  let forward (act:[]t -> []t) (l: i32)
               (training:bool)
               ((w, b): weights)
              (input:input) : (cache, output) = 
    let res      = lalg.matmul w (transpose input)
    let res_bias = transpose (map2 (\xr b' -> map (\x -> (R.(x + b'))) xr) res b)
    let res_act  = map (\x -> act x) (res_bias)
    let cache    = if training then (input, res_bias) else empty_cache
    in (replicate l cache, replicate l res_act)

  -- Backward propagation
  let backward (act: []t -> []t)
               (first_layer: bool)
               (apply_grads: apply_grad t)
               ((w, b): weights)
               (layer_caches: cache)
               (errors: error_in) : b_output =
    let l = length layer_caches
    let zero = R.from_fraction 0 1
    let fact = (R.from_fraction 1 1) R./ (R.from_fraction l 1)
    let average_sum_v [l][m] (matrix: [l][m]t): [m]t =
      util.scale_v (reduce util.add_v (replicate m (R.from_fraction 0 1)) matrix) fact
    let average_sum_matrix [l][m][n] (tensor: [l][m][n]t) : arr2d t=
      util.scale_matrix (reduce util.add_matrix (replicate m (replicate n zero)) tensor) fact
    -- Reduces weights from n to 1 layers
    let reduce_weights [l][m][n] (weights: [l]([m][n]t, [m]t)) : std_weights t = 
      let (ws, bs) = unzip weights
      let w        = average_sum_matrix ws
      let b        = average_sum_v bs
      in (w, b)

    let (errors, weights) = unzip (map (\i ->
	let (input, inp_w_bias) = layer_caches[i]
	let error    = errors[i]
	let deriv    = (map (\x -> act x) inp_w_bias)
	let delta    = transpose (util.hadamard_prod_2d error deriv)
	let w_grad   = lalg.matmul delta input
	let b_grad   = map (R.sum) delta
	let (w', b') = apply_grads (w,b) (w_grad, b_grad)

	--- Calc error to backprop to previous layer
	let error' =
	  if first_layer
	  then
	   empty_error
	  else
	   transpose (lalg.matmul (transpose w) delta)
	in (error', (w', b'))) (iota l))
    in (average_sum_matrix errors, reduce_weights weights)

  let init ((m,ns):input_params) (act:activations) (seed:i32) : replicate_nn =
    let w = w_init.gen_random_array_2d_xavier_uni (m,m) seed
    let b = map (\_ -> R.(i32 0)) (0..<m)
    let l = length ns
    in {forward  = forward act.f l,
	backward = backward act.fd,
	weights  = (w, b)}

}
