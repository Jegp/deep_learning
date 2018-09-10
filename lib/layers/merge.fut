import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "/futlib/linalg"


-- | Merges an array of layers
module merge (R:real) : layer_type with t = R.t
                                   with input_params = (i32, i32)
                                   with activations  = activation_func ([]R.t)
                                   with input        = arr3d R.t
                                   with weights      = [](std_weights R.t)
                                   with output       = arr2d R.t
                                   with cache        = [](arr2d R.t, arr2d R.t)
                                   with error_in     = arr2d R.t
                                   with error_out    = arr3d R.t = {

  type t            = R.t
  type input        = arr3d t
  type weights      = [](std_weights t)
  type output       = arr2d t
  type cache        = [](arr2d t, arr2d t)
  type error_in     = arr2d t
  type error_out    = arr3d t
  type b_output     = (error_out, []std_weights t)

  type input_params = (i32, i32)
  type activations  = activation_func ([]t)

  type merge_tp = NN input weights output
                     cache error_in error_out (apply_grad t)

  module lalg   = linalg R
  module util   = utility R
  module w_init = weight_initializer R

  let empty_cache:(arr2d t, arr2d t) = ([[]],[[]])
  let empty_error:error_in = [[]]

  -- Forward propagation
  let forward [n] (act:[]t -> []t)
               (training:bool)
               (weights:[n]std_weights t)
               (inputs:[n]arr2d t) : (cache, output) =
    let (caches, outputs) : (cache, []output) = 
      unzip (map2 (\(input) (w, b) ->
	let res      = lalg.matmul w (transpose input)
	let res_bias = transpose (map2 (\xr b' -> map (\x -> (R.(x + b'))) xr) res b)
	let res_act  = map (\x -> act x) (res_bias)
	let cache    = if training then (input, res_bias) else empty_cache
	in (cache, res_act)) inputs weights)
    in (caches, flatten outputs)

  -- Backward propagation
  let backward (act:[]t -> []t) (l: i32) (l_sz: i32)
               (first_layer:bool)
               (apply_grads:apply_grad t)
               (weights:weights)
               (layer_caches:cache)
               (error_list:error_in) : b_output =
    let split_errors : []error_in = unflatten l l_sz error_list

    let (errors, output_weights) : (error_out, []std_weights t) = 
      unzip (map3 (\(input, inp_w_bias) (w, b) error ->
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
	in (error', (w', b'))) layer_caches weights split_errors)
    in (errors, output_weights)

  let init ((l, m):input_params) (act:activations) (seed:i32) : merge_tp =
    let w = w_init.gen_random_array_2d_xavier_uni (m,m) seed
    let b = map (\_ -> R.(i32 0)) (0..<(l * m))
    in 
    {forward  = forward act.f,
     backward = backward act.fd l m,
     weights  = replicate l (w, b)}

}
