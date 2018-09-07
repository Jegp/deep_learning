import "../lib/deep_learning"
module dl = deep_learning f32

let seed = 1

let l1 = dl.layers.dense (784, 256) dl.nn.identity seed
let l2 = dl.layers.dense (256, 256) dl.nn.identity seed
let l3 = dl.layers.dense (256, 10) dl.nn.identity seed

let p1 = dl.layers.replicate (256, 1) dl.nn.identity seed
let m = dl.layers.merge (1, 256, 256) dl.nn.identity seed

--let nn0 = dl.nn.connect_layers l1 l2
--let nn = dl.nn.connect_layers nn0 l3

let nn0 = dl.nn.connect_layers l1 p1
let nn1 = dl.nn.connect_layers nn0 m
let nn  = dl.nn.connect_layers nn1 l3

let main [m] (input:[m][]dl.t) (labels:[m][]dl.t) =
  let train = 8000
  let validation = 2000
  let batch_size = 10
  let alpha = 0.1
  let nn' = dl.train.gradient_descent nn alpha
            input[:train] labels[:train]
            batch_size dl.loss.softmax_cross_entropy_with_logits
  in dl.nn.accuracy nn' input[train:train+validation]
     labels[train:train+validation] dl.nn.softmax dl.nn.argmax
