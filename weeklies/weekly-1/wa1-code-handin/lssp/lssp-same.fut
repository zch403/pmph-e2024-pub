-- Parallel Longest Satisfying Segment
--
entry mk_input1 (n:i64) : [20*n+20]i32 =
   let pattern = [-100i32, 10, 3, -1, 4, -1, 5, 1, 2, -100]
   let rep_pattern = replicate n pattern |> flatten
   let longest_segment = replicate 20 4
   in  (rep_pattern ++ longest_segment ++ rep_pattern) :> [20*n+20]i32

entry mk_input2 (n:i64) : [10*n+4]i32 =
   let pattern = [-100i32, 10, 3, -1, 4, -1, 5, 1, 1, -100]
   let rep_pattern = replicate n pattern |> flatten
   let longest_segment = replicate 4 0
   in  (rep_pattern ++ longest_segment) :> [10*n+4]i32

--
-- ==
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    5i32
-- }
--
-- compiled input {
--    [1i32, 1i32, 1i32, 1i32, 1i32, 1i32, 1i32, 1i32, 1i32, 1i32, 1i32, 1i32]
-- }
-- output {
--    12i32
-- }
--
-- compiled input {
--    [0i32, 1i32, 1i32, 0i32, 1i32, 1i32, 1i32, 1i32, 1i32, 1i32, 1i32, 0i32]
-- }
-- output {
--    7i32
-- }
--
-- script input { mk_input1 10000000i64 }
-- output { 20 }
--
-- script input { mk_input2 10000000i64 }
-- output { 4 }

import "lssp"
import "lssp-seq"

let main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
  --in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
