

single_valued = executable('single_valued', 'single_valued.cpp',
  include_directories:project_inc,
  link_with: project_lib,
  dependencies: main_deps
  )

pyro_construct = executable('pyro_construct', 'pyro_construct.cpp',
  include_directories:project_inc,
  link_with: project_lib,
  dependencies: main_deps
  )

cubic_construct = executable('cubic_construct', 'cubic_construct.cpp',
  include_directories:project_inc,
  link_with: project_lib,
  dependencies: main_deps
  )

pyro_fourier = executable('pyro_fourier', 'pyro_fourier.cpp',
  include_directories:project_inc,
  link_with: project_lib,
  dependencies: main_deps
  )

cubic_fourier = executable('cubic_fourier', 'cubic_fourier.cpp',
  include_directories:project_inc,
  link_with: project_lib,
  dependencies: main_deps
  )

snf_dims = executable('snf_dims', 'snf_dims.cpp',
  include_directories:project_inc,
  link_with: project_lib,
  dependencies: main_deps
  )


sublat_correlator = executable('sublat_correlator', 'sublat_correlator.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps
  )

planar_vs_3D = executable('planar_vs_3D', 'planar_vs_3D.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps
  )


basic_fourier = executable('basic_fourier', 'basic_fourier.cpp',
  include_directories: project_inc,
  link_with: project_lib,
  dependencies: main_deps
  )

test('snf_dims', snf_dims)
test('pyro_construct', pyro_construct, args: ['10', '0.5'])
test('cubic_construct', cubic_construct, args: ['10', '0.5'])
test('pyro_fourier', pyro_fourier)
test('cubic_fourier', cubic_fourier)
test('sublat_correlator', sublat_correlator)
