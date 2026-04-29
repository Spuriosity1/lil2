gtest_proj = subproject('gtest', default_options: ['default_library=static'])
gtest_dep = gtest_proj.get_variable('gtest_main_dep')

single_valued = executable('single_valued', 'single_valued.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps + [gtest_dep]
  )

pyro_construct = executable('pyro_construct', 'pyro_construct.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps + [gtest_dep]
  )

cubic_construct = executable('cubic_construct', 'cubic_construct.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps + [gtest_dep]
  )

pyro_fourier = executable('pyro_fourier', 'pyro_fourier.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps + [gtest_dep]
  )

cubic_fourier = executable('cubic_fourier', 'cubic_fourier.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps + [gtest_dep]
  )

snf_dims = executable('snf_dims', 'snf_dims.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps + [gtest_dep]
  )

sublat_correlator = executable('sublat_correlator', 'sublat_correlator.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps + [gtest_dep]
  )

planar_vs_3D = executable('planar_vs_3D', 'planar_vs_3D.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps + [gtest_dep]
  )

basic_fourier = executable('basic_fourier', 'basic_fourier.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps + [gtest_dep]
  )

test('snf_dims', snf_dims)
test('pyro_construct', pyro_construct)
test('cubic_construct', cubic_construct)
test('pyro_fourier', pyro_fourier)
test('cubic_fourier', cubic_fourier)
test('sublat_correlator', sublat_correlator)
test('basic_fourier', basic_fourier)
test('single_valued', single_valued)
