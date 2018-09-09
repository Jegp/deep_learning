import "../lib/deep_learning"
import "../lib/util"

module dl = deep_learning f64
module util = utility f64

-- ==
-- entry: par_network
-- input {[[1.0], [2.0]]
--
--        [[1.0], [2.0]]}
--
-- output { 1.0}

entry par_network input labels =
  let l2 = dl.layers.merge ([1], 1) dl.nn.identity 1
  let l1 = dl.layers.replicate (1, 1) dl.nn.identity 1
  let nn = dl.nn.connect_layers l1 l2

  let nn' = dl.train.gradient_descent nn 0.1 input labels 1 
	    dl.loss.softmax_cross_entropy_with_logits
  in dl.nn.accuracy nn' input
     labels dl.nn.softmax dl.nn.argmax


