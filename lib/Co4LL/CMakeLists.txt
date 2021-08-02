add_mlir_dialect_library(MLIRCo4LL
        Co4LLDialect.cpp
        Co4LLOps.cpp
        EmitXML.cpp
        BufAlloc.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Co4LL

        DEPENDS
        MLIRCo4LLOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
