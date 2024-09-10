-- Parallel Longest Satisfying Segment
--
entry mk_input1 (n:i64) : [20*n+20]i32 =
   let pattern = [-100i32, 10, 3, -1, 4, -1, 5, 1, 1, -100]
   let rep_pattern = replicate n pattern |> flatten
   let longest_segment = replicate 20 0
   in  (rep_pattern ++ longest_segment ++ rep_pattern) :> [20*n+20]i32

entry mk_input2 (n:i64) : [10*n+4]i32 =
   let pattern = [-100i32, 10, 3, -1, 4, -1, 5, 1, 1, -100]
   let rep_pattern = replicate n pattern |> flatten
   let longest_segment = replicate 4 0
   in  (rep_pattern ++ longest_segment) :> [10*n+4]i32
--
-- ==
-- compiled input {
--    [1i32, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }
-- output {
--    5
-- }
--
-- compiled input {
--    [1i32, -2, -2, 0, 0, 0, 1, 0, 3, 4, -6, 1]
-- }
-- output {
--    3
-- }
--
-- compiled input {
--    [1i32, -2, -2, 1, 1, 1, 1, 1, 3, 4, -6, 1]
-- }
-- output {
--    0
-- }
--
-- script input { mk_input1 10000000i64 }
-- output { 20 }
--
-- script input { mk_input2 10000000i64 }
-- output { 4 }

import "lssp-seq"
import "lssp"

type int = i32

let main (xs: []int) : int =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
