configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CO4_OPT_TEST_DEPENDS
        FileCheck count not
        co4-opt
        )

add_lit_testsuite(check-co4-opt "Running the co4-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CO4_OPT_TEST_DEPENDS}
        )
set_target_properties(check-co4-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(CO4_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CO4_OPT_TEST_DEPENDS})
