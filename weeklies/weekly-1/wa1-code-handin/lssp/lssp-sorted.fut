-- Parallel Longest Satisfying Segment
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

import "lssp"
import "lssp-seq"

type int = i32

let main (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x <= y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
