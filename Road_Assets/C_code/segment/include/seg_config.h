#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>


const int kGpuId = 0;
const int kNumClass = 50;
const int kInputH = 640;
const int kInputW = 640;
const float kNmsThresh = 0.45f;
const float kConfThresh = 0.45f;
const int kMaxNumOutputBbox = 1000;  // assume the box outputs no more than kMaxNumOutputBbox boxes that conf >= kNmsThresh;
const int kNumBoxElement =
        7 + 32;  // left, top, right, bottom, confidence, class, keepflag(whether drop when NMS), 32 masks

const std::string onnxFile = "/home/xianyang/Desktop/C++/DataResouse/model_file/onnx/citydet.onnx";
// const std::string trtFile = "./yolov8s.plan";
// const std::string testDataDir = "../images";  // 用于推理

// for FP16 mode
const bool bFP16Mode = true;
// for INT8 mode
const bool bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = "../calibrator";  // 存放用于 int8 量化校准的图像

const std::vector<std::string> vClassNames{
        "single_solid_line","single_dashed_line","double_solid_line","double_dashed_line","left_dashed_right_solid_line","left_solid_right_dashed_line",
        "fishbone_solid_line","fishbone_dashed_line","stop_line","straight_arrow","left_turn","right_turn","U_turn","merge_left","merge_right","left_right_turn",
        "straight_left_turn","straight_right_turn","left_uturn","crosswalk","yellow_net_line","diamond_line","parallel_line","guide_area","waiting_area",
        "pedestrian_waiting_area","parking_space","tree_trunk","pole","traffic_light","camera","gantry","bridge","bridge_pier","warning_signs","prohibition_signs",
        "guide_signs","guardrail","sound_insulation_tape","isolation_baffle","cement_wall","green_belt","anti_throw_net","kerbstone","manhole","electronic_screen","water_horse","crash_bucket","cone_bucket",
        "crash_bar",
};
// List of classes that should be drawn as bounding boxes
const std::set<std::string> bboxClasses = {
        "stop_line", "straight_arrow", "left_turn", "right_turn", "U_turn", "merge_left", "merge_right",
        "left_right_turn", "straight_left_turn", "straight_right_turn", "left_uturn",
        "crosswalk", "diamond_line", "parallel_line", "pedestrian_waiting_area", "parking_space", "tree_trunk", "pole",
        "traffic_light", "camera", "gantry", "bridge", "bridge_pier", "warning_signs",
        "prohibition_signs", "guide_signs", "manhole", "electronic_screen", "water_horse", "crash_bucket",
        "cone_bucket", "crash_bar"
};

// List of classes that should be drawn as masks
const std::set<std::string> maskClasses = {
        "single_solid_line", "single_dashed_line", "double_solid_line", "double_dashed_line",
        "left_dashed_right_solid_line", "left_solid_right_dashed_line",
        "fishbone_solid_line", "fishbone_dashed_line", "guardrail", "sound_insulation_tape", "cement_wall",
        "green_belt", "kerbstone", "yellow_net_line", "guide_area",
        "waiting_area", "anti_throw_net", "isolation_baffle"
};
const std::set<std::string> laneClasses = {
        "single_solid_line", "single_dashed_line", "double_solid_line", "double_dashed_line",
        "left_dashed_right_solid_line", "left_solid_right_dashed_line",
        "fishbone_solid_line", "fishbone_dashed_line", "guardrail", "sound_insulation_tape", "cement_wall",
        "kerbstone"
};
const std::set<std::string> other = {
        "yellow_net_line", "guide_area", "waiting_area", "anti_throw_net",
        "isolation_baffle"
};
#endif  // CONFIG_H
