
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputA.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputA.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputB.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputB.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputC.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputC.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputD.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputD.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputE.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputE.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/beam.pg
               ${CMAKE_CURRENT_BINARY_DIR}/beam.pg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/beam2b.pg
               ${CMAKE_CURRENT_BINARY_DIR}/beam2b.pg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material2block.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material2block.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName}_A ${SerialAlbany.exe} inputA.yaml)
add_test(${testName}_B ${SerialAlbany.exe} inputB.yaml)
add_test(${testName}_C ${SerialAlbany.exe} inputC.yaml)
add_test(${testName}_D ${SerialAlbany.exe} inputD.yaml)
add_test(${testName}_E ${SerialAlbany.exe} inputE.yaml)
endif()
