project('lattice_lib', ['cpp', 'c'],
  version: '2.0.0',
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
  snf_dep
  ]

project_lib=[]

# collect the sources
project_inc = include_directories('include')

subdir('src')

# build tests
if get_option('enable-tests')
  subdir('tests')
endif

# make project available
#pkg_mod = import('pkgconfig')
#pkg_mod.generate(libraries : lattice_indexing_lib,
#                 version : '1.1',
#                 name : 'liblatindex',
#                 description : 'A library for indexing r-chains')
#latlib_dep = declare_dependency(
#  link_with : lattice_indexing_lib,
#  include_directories : project_inc
#)
