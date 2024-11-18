//
// Created by xianyang on 24-8-20.
//

#include <string>
#include "xytools/tools.hpp"
#include <vector>

std::vector<int> tracksegClasses{0};

/*
/home/xianyang/xy/project/road_assets_new/models/models.plan

/home/xianyang/xy/project/road_assets_new/assets/test1.jpg
 * */
int main(int argc, char **argv) {
    // 判断输入
    bool isVideo = false;
    bool Image_or_Folder = false;
    bool use_camera = false;
    std::vector<std::string> imagePathList;
    std::string camera = "camera";
    if (argv[2] != camera) {
        const std::string path{argv[2]};
        if (IsFile(path)) {
            std::string suffix = path.substr(path.find_last_of('.') + 1);
            if (suffix == "jpg" || suffix == "jpeg" || suffix == "png") {
                imagePathList.push_back(path);
                Image_or_Folder = true;
            } else if (suffix == "mp4" || suffix == "avi") {
                isVideo = true;
            } else {
                printf("suffix %s is wrong !!!\n", suffix.c_str());
                std::abort();
            }
        } else if (IsFolder(path)) {
            cv::glob(path + "/*.jpg", imagePathList);
            Image_or_Folder = true;
        }
    } else {
        use_camera = true;
    }

    const std::string engine_file{argv[1]};

    if (isVideo) {
        std::string output_mp4 = "../output/res.mp4"; //保存路径
        detect_mp4(engine_file, argv[2], tracksegClasses, output_mp4);
    }
    if (Image_or_Folder) {
        std::string output_directory = "/home/xianyang/xy/project/road_assets_new/output/"; //保存路径
        detect_img(engine_file, imagePathList, output_directory);
    }
    if (use_camera) {

    }

    return 0;
}
