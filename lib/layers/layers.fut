import "../nn_types"
import "layer_type"
import "dense"
import "conv2d"
import "flatten"
import "max_pooling"
import "merge"
import "replicate"

module type layers = {

  type t

  --- Layer types
  type dense_tp =
    NN (arr2d t) (arr2d t,arr1d t) (arr2d t)
       ((arr2d t, arr2d t)) (arr2d t) (arr2d t) (apply_grad t)
  type replicate_tp =
    NN (arr2d t) (arr2d t,arr1d t) (arr3d t)
       ((arr2d t, arr2d t)) (arr3d t) (arr2d t) (apply_grad t)
  type conv2d_tp  =
    NN (arr4d t) (arr2d t,arr1d t) (arr4d t)
       (dims3d, arr3d t, arr4d t) (arr4d t) (arr4d t) (apply_grad t)
  type max_pooling_tp  =
    NN (arr4d t) () (arr4d t) (arr4d (i32)) (arr4d t) (arr4d t) (apply_grad t)
  type merge_tp =
    NN ([]arr2d t) ((arr2d t,arr1d t)) (arr2d t)
       ([](arr2d t, arr2d t)) (arr2d t) ([]arr2d t) (apply_grad t)
  type flatten_tp  =
    NN (arr4d t) () (arr2d t) dims3d (arr2d t) (arr4d t) (apply_grad t)
  

  -- Simple wrappers for each layer type
  val dense: (i32, i32) -> (activation_func ([]t)) ->  i32 -> dense_tp
  val conv2d: (i32, i32, i32, i32) -> (activation_func ([]t)) -> i32 -> conv2d_tp
  val max_pooling2d: (i32, i32) -> max_pooling_tp
  val merge: ([]i32, i32) -> (activation_func ([]t)) -> i32 -> merge_tp
  val flatten: flatten_tp
  val replicate: (i32, i32) -> (activation_func ([]t)) -> i32 -> replicate_tp
}

module layers_coll (R:real): layers with t = R.t = {

  type t = R.t

  --- Layer types
  type dense_tp       =
      NN (arr2d t) (arr2d t,arr1d t) (arr2d t)
         ((arr2d t, arr2d t)) (arr2d t) (arr2d t) (apply_grad t)
  type replicate_tp =
    NN (arr2d t) (arr2d t,arr1d t) (arr3d t)
       ((arr2d t, arr2d t)) (arr3d t) (arr2d t) (apply_grad t)
  type merge_tp =
    NN ([]arr2d t) ((arr2d t,arr1d t)) (arr2d t)
       ([](arr2d t, arr2d t)) (arr2d t) ([]arr2d t) (apply_grad t)
  type conv2d_tp      =
      NN (arr4d t) (arr2d t,arr1d t) (arr4d t)
         (dims3d, arr3d t, arr4d t) (arr4d t) (arr4d t) (apply_grad t)
  type max_pooling_tp =
      NN (arr4d t) () (arr4d t) (arr4d (i32)) (arr4d t) (arr4d t) (apply_grad t)
  type flatten_tp     =
      NN (arr4d t) () (arr2d t) (dims3d) (arr2d t) (arr4d t) (apply_grad t)

  module merge_layer   = merge R
  module dense_layer   = dense R
  module conv2d_layer  = conv2d R
  module flatten_layer = flatten R
  module maxpool_layer = max_pooling_2d R
  module replicate_layer = replicate R

  let dense ((m,n):dense_layer.input_params)
            (act_id:dense_layer.activations)
            (seed:i32)  =
    dense_layer.init (m,n) act_id seed

  let replicate ((m,ns):replicate_layer.input_params)
		(act_id:replicate_layer.activations)
		(seed: i32) =
    replicate_layer.init(m, ns) act_id seed

  let conv2d (params:conv2d_layer.input_params)
             (act:conv2d_layer.activations)
             (seed:i32) =
    conv2d_layer.init params act seed

  let flatten =
    flatten_layer.init () () 0

  let max_pooling2d (params:maxpool_layer.input_params) =
    maxpool_layer.init params () 0

  let merge (params: merge_layer.input_params)
            (act_id: merge_layer.activations)
	    (seed: i32) =
    merge_layer.init params act_id seed

}
