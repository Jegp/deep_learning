import "../nn_types"


module type layer_type = {

  type t

  type input_params
  type ^activations

  type input
  type output
  type weights_in
  type weights_out
  type error_in
  type error_out
  type cache_in
  type cache_out

  --- Initialize layer given input params
  val init: input_params -> activations -> i32 ->
         NN input weights_in weights_out output cache_in cache_out error_in error_out (apply_grad t)

}
