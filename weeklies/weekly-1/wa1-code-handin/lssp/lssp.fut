-----------------------------------------
-- Parallel Longest Satisfying Segment --
-- Longest Satisfying Segment          --
-- ASSIGNMENT 1: fill in the blanks    --
--       See lecture notes             --
-- pred1 x   = p [x]                   --
-- pred2 x y = p [x,y]                 --
-----------------------------------------

-- use `max a b` to compute the max between i32 values a and b.
let max = i32.max

-- The sextuple (6-tuple) used to hold meta-data for the LSS computation.
type lss_t = (i32, i32, i32, i32, i32, i32)

let lss_redOp (pred2: i32 -> i32 -> bool)
              ((x_lss, x_lis, x_lcs, x_len, x_first, x_last): lss_t)
              ((y_lss, y_lis, y_lcs, y_len, y_first, y_last): lss_t)
                : lss_t =

  -- Weekly 1, task 2 -- LSSP
  -- TODO: fill in the 5 missing values (indicated by `???`).
  let segments_connect = 
    if (x_len == 0 || y_len == 0) then true 
    else pred2 x_last y_first 

  let new_lss = if segments_connect then (max (x_lcs+y_lis) (max x_lss y_lss)) else  max x_lss y_lss
  let new_lis = if x_lss == x_len && segments_connect then x_lss+y_lis else x_lis
  let new_lcs = if y_lss == y_len && segments_connect then x_lcs+y_lss else y_lcs
  let new_len = x_len + y_len

  let new_first = if x_len == 0 then y_first else x_first
  let new_last  = if y_len == 0 then x_last else y_last
   in (new_lss, new_lis, new_lcs, new_len, new_first, new_last)

let lss_mapOp (pred1: i32 -> bool) (x: i32) : lss_t =
  let xmatch = i32.bool (pred1 x)
   in (xmatch, xmatch, xmatch, 1, x, x)

let lssp (pred1: i32 -> bool)
         (pred2: i32 -> i32 -> bool)
         (xs: []i32)
           : i32 =
  map (lss_mapOp pred1) xs
  |> reduce (lss_redOp pred2) (0, 0, 0, 0, 0, 0)
  |> (.0)
