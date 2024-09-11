-- Parallel Longest Satisfying Segment
--
entry mk_input1 (n:i64) : [20*n+20]i32 =
   let pattern = [-100i32, 10, 3, -1, 4, -1, 5, 1, 1, -100]
   let rep_pattern = replicate n pattern |> flatten
   let longest_segment = iota 20 |> map i32.i64
   in  (rep_pattern ++ longest_segment ++ rep_pattern) :> [20*n+20]i32

entry mk_input2 (n:i64) : [10*n+4]i32 =
   let pattern = [-100i32, 10, 3, -1, 4, -1, 5, 1, 1, -100]
   let rep_pattern = replicate n pattern |> flatten
   let longest_segment = iota 4 |> map i32.i64
   in  (rep_pattern ++ longest_segment) :> [10*n+4]i32
--
-- ==
-- compiled input {
--    [1, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }  
-- output { 
--    9
-- }
--
-- compiled input {
--    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
-- }  
-- output { 
--    12
-- }
--
-- compiled input {
--    [1, 0, 4, 8, 16, 32, 0, 128, 256, 512, 0, 2048]
-- }  
-- output { 
--    5
-- }
--
-- script input { mk_input1 10000000i64 }
-- output { 21 }
--
-- script input { mk_input2 10000000i64 }
-- output { 5 }

import "lssp"
import "lssp-seq"

type int = i32

let main (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x <= y)
  --in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
