import "../lib/deep_learning"
module dl = deep_learning f64

let seed = 1

let l1 = dl.layers.dense (784, 256) dl.nn.sigmoid seed
let l2 = dl.layers.dense (256, 256) dl.nn.sigmoid seed
let l3 = dl.layers.dense (256, 10) dl.nn.sigmoid seed

let nn1 = dl.nn.connect_layers l1 l2
let nn  = dl.nn.connect_layers nn1 l3

-- ==
--
-- tags { futhark-opencl }
-- input @ ../data/mnist_10000.bindata

let main [m] (input: [m][]dl.t) (labels: [m][]dl.t) =
  let n = 6400
  let batch_size = 128
  let alpha = 0.01
  let nn' = dl.train.gradient_descent nn alpha
            input[:n] labels[:n]
            batch_size dl.loss.softmax_cross_entropy_with_logits
  in nn'.weights
