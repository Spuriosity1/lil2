project('lattice_lib', ['cpp', 'c'],
  version: '2.2.1',
  meson_version : '>= 1.5.0',
  default_options : [ 'warning_level=3', 'buildtype=release', 'cpp_std=c++2a' ]
)

project_description = 'Indexing library for periodic lattices'

cmake = import('cmake')


# build dependencies
snf_project = subproject('smith_normal_form', default_options: [
  'enable-tests=false'
  ])
snf_dep = snf_project.get_variable('snf_dep')


# import the #define directives
cpp_proj_arguments = snf_project.get_variable('cpp_proj_arguments')
add_project_arguments(cpp_proj_arguments, language : 'cpp')


fftw_dep = dependency('fftw3', required: true)

#hdf5_dep = dependency('hdf5', required: true, language: 'c')

main_deps = [
  dependency('nlohmann_json', required: true),
  fftw_dep,
  snf_dep,
  dependency('argparse', required: true)
  ]

project_lib=[]

# collect the sources
project_inc = include_directories(['include'])

# install headers
subdir('include')
# build the library 
subdir('src')

# build tests
if get_option('enable-tests')
  subdir('tests')
endif

# make project available
pkg_mod = import('pkgconfig')
pkg_mod.generate(libraries : lil_lib,
                 name : 'latlib',
                 description : 'A library for building periodic arrays of objects cleanly',
  version: meson.project_version(),
  filebase: 'lattice_lib', 
  subdirs: 'lattice_lib'
  )


latlib_dep = declare_dependency(
  link_with : lil_lib,
  dependencies: main_deps,
  include_directories : project_inc
)
