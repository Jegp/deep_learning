import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "/futlib/linalg"


-- | Merges an array of layers
module merge (R:real) : layer_type with t = R.t
                                   with input_params = ([]i32, i32)
                                   with activations  = activation_func ([]R.t)
                                   with input        = []arr2d R.t
                                   with weights      = []std_weights R.t
                                   with output       = arr2d R.t
                                   with cache        = (arr2d R.t, arr2d R.t)
                                   with error_in     = []arr2d R.t
                                   with error_out    = arr2d R.t = {

  type t            = R.t
  type input        = []arr2d t
  type weights      = []std_weights t
  type output       = arr2d t
  type cache        = (arr2d t, arr2d t)
  type error_in     = []arr2d t
  type error_out    = arr2d t
  type b_output     = (error_out, weights)

  type input_params = ([]i32, i32)
  type activations  = activation_func ([]t)

  type merge_tp = NN input weights output
                     cache error_in error_out (apply_grad t)

  module lalg   = linalg R
  module util   = utility R
  module w_init = weight_initializer R

  let empty_cache:cache= ([[]],[[]])
  let empty_error:error_out = [[]]

  -- Forward propagation
  let forward  (act:[]t -> []t) (l: i32)
               (training:bool)
               (layer_weights:weights)
               (inputs:input) : (cache, output) =
    let (caches, outputs) = unzip (map (\i ->
        let (w, b)   = layer_weights[i]
	let input    = inputs[i]
	let res      = lalg.matmul w (transpose input)
	let res_bias = transpose (map2 (\xr b' -> map (\x -> (R.(x + b'))) xr) res b)
	let res_act  = map (\x -> act x) (res_bias)
	let cache    = if training then (input, res_bias) else empty_cache
	in (cache, res_act)) (iota l))
    let (c1, c2) = unzip caches
    in ((flatten c1, flatten c2), flatten outputs)

  -- Backward propagation
  let backward (act:[]t -> []t) (l:i32)
               (first_layer:bool)
               (apply_grads:apply_grad t)
               (layer_weights:weights)
               ((input, inp_w_bias):cache)
               (errors:error_in) : b_output =
    let (errors, output_weights) = unzip (map (\i ->
        let (w, b)   = layer_weights[i]
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
    in (flatten errors, output_weights)

  let init ((ms,n):input_params) (act:activations) (seed:i32) : merge_tp =
    let l = length ms
    let weights = map (\m -> 
	let w = w_init.gen_random_array_2d_xavier_uni (m,n) seed
	let b = map (\_ -> R.(i32 0)) (0..<n)
	in (w, b)) (iota l)
    in
    {forward  = forward act.f l,
     backward = backward act.fd l,
     weights  = weights}

}
