add_mlir_dialect_library(MLIRCo4HL
        Co4HLDialect.cpp
        Co4HLOps.cpp
        Lower.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Co4HL

        DEPENDS
        MLIRCo4HLOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
