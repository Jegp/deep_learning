-- | Network types
type forwards   'input 'w 'output 'cache = bool -> w -> input -> (cache, output)
type backwards  'c 'w_in 'w_out  'err_in  'err_out '^u = bool -> u -> w_out -> c -> err_in  -> (err_out, w_in)

type NN 'input 'w_in 'w_out 'output 'c_in 'c_out 'e_in 'e_out '^u =
               { forward : forwards input w_in output c_out,
                 backward: backwards c_out w_in w_out e_in e_out u,
                 weights : w_out}

--- Commonly used types
type arr1d 't = []t
type arr2d 't = [][]t
type arr3d 't = [][][]t
type arr4d 't = [][][][]t

type dims2d  = (i32, i32)
type dims3d  = (i32, i32, i32)

--- The 'standard' weight definition
--- used by optimizers
type std_weights 't = ([][]t, []t)
type apply_grad 't  = std_weights t -> std_weights t -> std_weights t

--- Function pairs
--- Denotes a function and it's derivative
type activation_func 'o = {f:o -> o, fd:o -> o}
type loss_func 'o  't   = {f:o -> o -> t, fd:o -> o -> o}
