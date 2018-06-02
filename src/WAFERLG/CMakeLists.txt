##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

SET(WAFERLG_DIR ${Albany_SOURCE_DIR}/src/WAFERLG)

set(waferlg-problem-sources
  "${WAFERLG_DIR}/problems/WAFERLGThermoMechanics.cpp"
)
set(waferlg-problem-headers
  "${WAFERLG_DIR}/problems/WAFERLGThermoMechanics.hpp"
)

set(waferlg-evaluator-sources
  "${WAFERLG_DIR}/evaluators/WAFERLG_Time.cpp"
  "${WAFERLG_DIR}/evaluators/Energy_Dot.cpp"
  "${WAFERLG_DIR}/evaluators/Laser.cpp"
  "${WAFERLG_DIR}/evaluators/Laser_Source.cpp"
  "${WAFERLG_DIR}/evaluators/Phase_Residual.cpp"
  "${WAFERLG_DIR}/evaluators/Phase_Source.cpp"
  "${WAFERLG_DIR}/evaluators/Phi1.cpp"
  "${WAFERLG_DIR}/evaluators/Phi2.cpp"
  "${WAFERLG_DIR}/evaluators/Psi1.cpp"
  "${WAFERLG_DIR}/evaluators/Psi2.cpp"
  "${WAFERLG_DIR}/evaluators/rho_Cp.cpp"
  "${WAFERLG_DIR}/evaluators/ThCond.cpp"
)
set(waferlg-evaluator-headers
  "${WAFERLG_DIR}/evaluators/WAFERLG_Time_Def.hpp"
  "${WAFERLG_DIR}/evaluators/WAFERLG_Time.hpp"
  "${WAFERLG_DIR}/evaluators/Energy_Dot_Def.hpp"
  "${WAFERLG_DIR}/evaluators/Energy_Dot.hpp"
  "${WAFERLG_DIR}/evaluators/Laser.hpp"
  "${WAFERLG_DIR}/evaluators/Laser_Source_Def.hpp"
  "${WAFERLG_DIR}/evaluators/Laser_Source.hpp"
  "${WAFERLG_DIR}/evaluators/Phase_Residual_Def.hpp"
  "${WAFERLG_DIR}/evaluators/Phase_Residual.hpp"
  "${WAFERLG_DIR}/evaluators/Phase_Source_Def.hpp"
  "${WAFERLG_DIR}/evaluators/Phase_Source.hpp"
  "${WAFERLG_DIR}/evaluators/Phi1_Def.hpp"
  "${WAFERLG_DIR}/evaluators/Phi2_Def.hpp"
  "${WAFERLG_DIR}/evaluators/Phi1.hpp"
  "${WAFERLG_DIR}/evaluators/Phi2.hpp"
  "${WAFERLG_DIR}/evaluators/Psi1_Def.hpp"
  "${WAFERLG_DIR}/evaluators/Psi2_Def.hpp"
  "${WAFERLG_DIR}/evaluators/Psi1.hpp"
  "${WAFERLG_DIR}/evaluators/Psi2.hpp"
  "${WAFERLG_DIR}/evaluators/rho_Cp_Def.hpp"
  "${WAFERLG_DIR}/evaluators/rho_Cp.hpp"
  "${WAFERLG_DIR}/evaluators/ThCond_Def.hpp"
  "${WAFERLG_DIR}/evaluators/ThCond.hpp"
)

set(SOURCES
  ${waferlg-problem-sources}
  ${waferlg-evaluator-sources}
  )

set(HEADERS
  ${waferlg-problem-headers}
  ${waferlg-evaluator-headers}
 )

set(INCLUDE_DIRS
  ${Trilinos_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src
  ${Albany_SOURCE_DIR}/src/evaluators
  ${Albany_SOURCE_DIR}/src/problems 
  ${Albany_SOURCE_DIR}/src/disc/pumi
  ${Albany_SOURCE_DIR}/src/adapt
  ${WAFERLG_DIR}/problems
  ${WAFERLG_DIR}/evaluators
  ${ALBANY_PUMI_INCLUDE_DIRS} 
  ${Trilinos_TPL_INCLUDE_DIRS}
)

include_directories(${INCLUDE_DIRS})
link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})
add_library(WAFERLG ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
set_target_properties(WAFERLG PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF ( INSTALL_ALBANY )
  install(TARGETS WAFERLG EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
