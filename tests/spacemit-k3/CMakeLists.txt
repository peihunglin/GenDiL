if(GENDIL_ENABLE_K3_IME_EXPERIMENTS)
  gendil_add_test(ime-pilot-correctness ime_pilot_correctness.cpp)
endif()

if(GENDIL_ENABLE_K3_IME_NATIVE)
  gendil_add_test(
    ime-interpolation-correctness
    ime_interpolation_correctness.cpp
  )
endif()

if(GENDIL_ENABLE_K3_FP16_BASELINE)
  gendil_add_test(fp16-baseline-correctness fp16_baseline_correctness.cpp)
endif()
