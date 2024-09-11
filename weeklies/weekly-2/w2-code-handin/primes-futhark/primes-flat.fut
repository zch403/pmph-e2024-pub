-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 }

-- output @ ref10000000.out
let sgmScan [n] 't
            (op: t -> t -> t)
            (ne: t)
            (flags: [n]bool)
            (vals: [n]t)
            : [n]t =
  scan (\(f1, v1) (f2, v2) -> (f1 || f2, if f2 then v2 else op v1 v2))
       (false, ne)
       (zip flags vals)
  |> unzip
  |> (.1)

let exclusiveScan [n] 't
            (op: t -> t -> t)
            (ne: t)
            (vals: [n]t)
            : [n]t =
  let nVals = map (\i -> if i==0 then ne else vals[i-1]) (iota n)
  in scan op ne nVals

let primesFlat (n : i64) : []i64 =
  let sq_primes   = [2i64, 3i64, 5i64, 7i64]
  let len  = 8i64
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i64 bounds 
      let len = if n / len < len then n else len*len

      let mult_lens = map (\ p -> (len / p) - 1 ) sq_primes -- [14, 9, 5, 3]
      let flat_size = reduce (+) 0 mult_lens -- 31

      --------------------------------------------------------------
      -- The current iteration knows the primes <= 'len', 
      --  based on which it will compute the primes <= 'len*len'
      -- ToDo: replace the dummy code below with the flat-parallel
      --       code that is equivalent with the nested-parallel one:
      --
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite
      --
      -- Your code should compute the correct `not_primes`.
      -- Please look at the lecture slides L2-Flattening.pdf to find
      --  the normalized nested-parallel version.
      -- Note that the scalar computation `mm1 = (len / p) - 1' has
      --  already been distributed and the result is stored in "mult_lens",
      --  where `p \in sq_primes`.
      -- Also note that `not_primes` has flat length equal to `flat_size`
      --  and the shape of `composite` is `mult_lens`. 
      
      -- let sqrt_primes = primesOpt (sqrt (fromIntegral n)) 
      -- let nested = map (\p ->
      --   let m    = n / p           in
      --   let mm1  = m - 1           in
      --   let iot  = iota mm1        in
      --   let twom = map (+2) iot    in
      --   let rp   = replicate mm1 p in 
      --   map (\(j,p) -> j*p) (zip twom rp)
      --               ) sq_primes

      -- let nested = map (\p -> p) sq_primes
      -- let not_primes  = reduce (++) [] nested -- ignore, already flat
      let escan_mult_lens = (exclusiveScan (+) 0 mult_lens) -- [0, 14, 23, 28]
      let flags = scatter (replicate flat_size false) escan_mult_lens (map (\_ -> true) escan_mult_lens) -- [true, false, ..., true(14), false, ..., true(23), false, ..., true(28), false, ... ]
      let tmp1 = replicate flat_size 1 -- [1] length 31
      let tmp2 = sgmScan (+) 0 flags tmp1 -- [1..14, 1..9, 1..5, 1..3]
      let tmp3 = map (+1) tmp2 -- [2..15, 2..10, 2..6, 2..4]
      let tmp6 = replicate flat_size 0 -- [0] length 31
      let tmp4 = scatter tmp6 escan_mult_lens sq_primes -- [2, 0, ..., 3, 0, ..., 5, 0, ..., 7, 0, ... ]
      let tmp5 = sgmScan (\acc _ -> acc) 0 flags tmp4 -- [2, 2, ..., 3, 3, ..., 5, 5, ..., 7, 7, ... ]
      let not_primes = map2 (\nr p -> nr*p) tmp3 tmp5 -- [4, 6, ..., 6, 9, ..., 10, 15, ..., 14, 21, ... ]

      -- If not_primes is correctly computed, then the remaining
      -- code is correct and will do the job of computing the prime
      -- numbers up to n!
      --------------------------------------------------------------
      --------------------------------------------------------------

       let zero_array = replicate flat_size 0i8
       let mostly_ones= map (\ x -> if x > 1 then 1i8 else 0i8) (iota (len+1))
       let prime_flags= scatter mostly_ones not_primes zero_array
       let sq_primes = filter (\i-> (i > 1i64) && (i <= n) && (prime_flags[i] > 0i8))
                              (0...len)

       in  (sq_primes, len)

  in sq_primes

-- RUN a big test with:
--   $ futhark cuda primes-flat.fut
--   $ echo "10000000i64" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
-- or simply use futhark bench, i.e.,
--   $ futhark bench --backend=cuda primes-flat.fut
let main (n : i64) : []i64 = primesFlat n
