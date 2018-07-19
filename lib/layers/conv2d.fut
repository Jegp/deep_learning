import "layer_type"
import "../nn_types"
import "../util"
import "../random_gen"
import "/futlib/linalg"

module conv2d (R:real) : layer with t = R.t
                               with input_params = (i32,i32, i32, i32)
                               with activations  = f_pair_1d R.t
                               with input        = arr4d  R.t
                               with weights      = (arr2d R.t,arr1d R.t)
                               with output       = arr4d R.t
                               with cache        = (dims3d, arr4d R.t, arr4d R.t)
                               with error_in     = arr4d R.t
                               with error_out    = arr4d R.t = {

  type t = R.t
  type input        = arr4d  t
  type weights      = std_weights t
  type output       = arr4d t
  type dims         = dims3d
  type cache        = (dims, arr4d t, arr4d t)
  type error_in     = arr4d t
  type error_out    = arr4d t
  type b_output     = (error_out, weights)

  type input_params = (i32 ,i32, i32, i32)
  type activations  = f_pair_1d R.t

  module lalg   = linalg R
  module util   = utility R
  module random = normal_random_array R

  let zero_dims: dims = (0,0,0)
  let empty_cache: cache = (zero_dims, [[[]]], [[[[]]]])
  let empty_error: error_out = [[[[]]]]

  let calc_index (stride:i32) ((m,n):(i32, i32)): [](i32, i32) =
    let row_index = map (\i -> i * stride) (0..<m)
    let col_index = map (\i -> i * stride) (0..<n)
    in flatten (map (\i -> map (\j -> (i,j) ) row_index) col_index)

  let add_padding (padding:i32) (X:arr2d t) : arr2d t =
    let height   = length X    + padding * 2
    let width    = length X[0] + padding * 2
    let tot_elem = width * height
    let index    = (flatten (map (\i -> (map (\j -> (i,j)) (0..<length X))) (0..<length X[0])))
    let offsets  = map (\(i,j) -> padding*width + padding + width * i + j) index
    let retval   = scatter (map (\_ -> R.(i32 0)) (0..<tot_elem)) (offsets) (flatten X)
    in unflatten height width retval

  let im2col (x:arr3d t) ((w_m, w_n):(i32, i32)) (idx:arr1d  (i32, i32)) : arr2d t =
    unsafe transpose (map (\(i,j) ->
                           flatten (map (\layer -> flatten layer[i:i+w_m, j:j+w_n]) x)) idx)

  let dot_prod (x:[]t) (y:[]t) =
    reduce (R.+) R.(i32 0) (map2 (\x' y'  -> R.(x' * y'))  x y)


  let dot_prod_2d (X:[][]t) (Y:[][]t) =
    let tmp = map2 (\x y -> map2 (\x' y' -> R.(x' * y')) x y ) X Y
    in reduce (R.+) R.(i32 0) (flatten tmp)

  let convolve3d_1 (x:[][][]t) (w:[][]t)  ((w_m, w_n):(i32, i32)) (idx:[](i32, i32))  =
    unsafe (map (\(i,j) ->
          let slice = (map (\layer -> flatten layer[i:i+w_m, j:j+w_n]) x)
           in dot_prod_2d slice w) idx)

  let convolve2d_1 (x:[][]t) (w:[][]t)  ((w_m, w_n):(i32, i32)) (idx:[](i32, i32))  =
    unsafe (map (\(i,j) ->
          let slice = x[i:i+w_m, j:j+w_n]
           in dot_prod_2d slice w) idx)


  let convolve3d (x:[][][]t) (w:[]t)  ((w_m, w_n):(i32, i32)) (idx:[](i32, i32))  =
    unsafe (map (\(i,j) ->
          let slice = flatten (map (\layer -> flatten layer[i:i+w_m, j:j+w_n]) x)
           in dot_prod slice w) idx)

  let convolve2d (x:[][]t) (w:[]t)  ((w_m, w_n):(i32, i32)) (idx:[](i32, i32))  =
    unsafe (map (\(i,j) ->
          let slice =flatten  x[i:i+w_m, j:j+w_n]
           in dot_prod slice w) idx)



  let forward (act:[]t -> []t) ((w_m, w_n):(i32, i32)) (stride:i32)
                               (training:bool) ((w,b):weights) (input:input) : (cache, output) =
    let (x_p, x_m, x_n) = (length input[0], length input[0,0], length input[0,0,0])
    let (out_m, out_n)  = (((x_m - w_m)/ stride) + 1, ((x_n - w_n)/stride) + 1)
    let indexs          = calc_index stride (out_m, out_n)
    let res             = map (\image -> map (\w' ->  convolve3d image w' (w_m,w_n) indexs) w ) input
    let res_bias        = map (\image -> map2 (\b' layer -> map (\x -> R.(x + b')) layer) b image) res
    let res_act         = map (\image -> map (\layer -> act layer) image ) res_bias
    let res_bias'       = map (\img   -> map (\x -> unflatten out_m out_n x) img) res_bias
    let cache           = if training then ((x_p, x_m, x_n), input, res_bias') else empty_cache
    in (cache, map (\inp -> map (\x -> unflatten out_m out_n x) inp) res_act)



  let backward (act:[]t->[]t) (k:i32) (stride:i32) (first_layer:bool) ((w,_): weights)
                                      ((dims, input0, input1):cache) (error:error_in) : b_output =

    let (x_p, x_m, x_n) = dims
    let res_deriv       = map (\image -> map (\layer ->  map (\row -> act row) layer) image) input1
    let delta           = util.mult_matrix_4d error res_deriv
    -- let delta_flat      = map (\img -> map (\layer -> flatten layer ) img ) delta
    let delta_ix        = calc_index stride (k,k)
    let grads_w         = map2 (\img delta -> map (\d' -> map (\img' -> convolve2d_1 img' d' (2,2) delta_ix ) img ) delta)  input0 delta
    let grad_w_flat     = map (\x -> flatten (map (\x' -> flatten x') x)) grads_w
    let grad_w          = unflatten (length w) (length w[0]) (map (R.sum) (transpose grad_w_flat))
    let grads_b         = map (\img -> map (\d' ->  R.sum (flatten d')) img ) delta
    let grad_b          = (map (R.sum) (transpose (grads_b)))


    --- Calc error for previous layer ----
    let kz           = k * k
    let w_offsets    = map (\i -> i * kz) (0..<x_p)
    let w_split      = map (\i -> flatten ((map (\r -> reverse (r[i:i+kz])) w))) w_offsets
    let delta_padded = map (\d -> map (\x -> add_padding (k-1) x) d) delta
    let delta_ix     = calc_index stride (x_m, x_n)
    let error        = map (\img -> map (\w' -> convolve3d img w' (k,k) delta_ix ) w_split) delta_padded
    let error'       = map (\img -> map (\d' -> unflatten x_m x_n d') img) error

    in

    (error', (grad_w,grad_b))

    -- let delta_flat      = map (\img -> map (\x -> flatten x) img) delta
    -- let grads_w         = map2 (\i d -> transpose (lalg.matmul i (transpose d))) input0 delta_flat
    -- let grad_w          = map (\d -> map (R.sum) (transpose d)) (transpose grads_w)

    -- let grads_b         = map (\img -> map (\layer -> R.sum (flatten layer) ) img) delta
    -- let grad_b          = map (R.sum) (transpose grads_b)

    -- --- Calc error for previous layer ----
    -- let error' =
    --   if first_layer
    --   then
    --     empty_error
    --   else
    --     let kz           = k * k
    --     let w_offsets    = map (\i -> i * kz) (0..<x_p)
    --     let w_split      = map (\i -> flatten ((map (\r -> reverse (r[i:i+kz])) w))) w_offsets
    --     let delta_padded = map (\d -> map (\x -> add_padding (k-1) x) d) delta
    --     let delta_ix     = calc_index stride (x_m, x_n)
    --     let delta_col    = map (\d -> im2col d (k,k) delta_ix) delta_padded
    --     let error        = map (\x -> lalg.matmul (w_split) (x)) delta_col
    --     in map (\img -> map (\x -> (unflatten x_m x_n x)) img ) error
    -- in (error' , (grad_w,grad_b))

  let update (f:apply_grad t) (w:weights) (wg:weights) =
    f w wg

  let init ((filters, kernel, stride, depth):input_params)  (act:activations)  (seed: i32)  =
    let w: arr2d  t  = (random.gen_random_array_2d_w_scaling ((kernel* kernel * depth), filters) seed)
    let b: arr1d  t  = map (\_ -> R.(i32 0)) (0..<filters)
   in
    (forward act.1 (kernel,kernel) stride,
     backward act.2 kernel stride,
     update,
     (w,b))
}

    -- let image_matrix    = map (\image -> im2col image (w_m,w_n) indexs) input
    -- let res             = map (\image -> (lalg.matmul w image) ) image_matrix
    -- let res_bias        = map (\image ->
    --                            map2 (\layer b' -> map (\x -> R.(x + b')) layer) image b) res
    -- let res_bias'       = map (\inp ->
    --                            map (\x -> unflatten out_m out_n x) inp) res_bias
    -- let res_act         = map (\image ->
    --                            map (\layer -> act layer ) image) res_bias