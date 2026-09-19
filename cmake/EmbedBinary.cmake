if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "EmbedBinary.cmake requires INPUT and OUTPUT")
endif()

file(READ "${INPUT}" contents HEX)
string(REGEX REPLACE "(..)" "0x\\1," contents "${contents}")
file(
  WRITE "${OUTPUT}"
  "static constexpr unsigned char kLightRocrCopyKernelHsaco[] = {${contents}};\n"
)
