#ifndef _DATALOADER_HPP
#define _DATALOADER_HPP

#include <fstream>
#include <vector>
#include "Utils.h"

class DataLoader {
    public:
        DataLoader(std::string dataDirPath) {
            std::cout << "DataLoader >> Initializing ..." << std::endl;
            std::cout << "DataLoader >> Reading Depth and RGB Frame ..." << std::endl;
            m_dataDirPath = dataDirPath;
            std::ifstream readStream = std::ifstream(m_dataDirPath + "rgbd_assoc_poses.txt");
            if (!readStream.is_open()) {
                throw std::invalid_argument("No such File " + m_dataDirPath);
            }

            int lineCounter = 0;

            for (std::string line; std::getline(readStream, line);) {
                std::vector<std::string> lineComponents = split(line, " +");
                std::string depthFramePath = lineComponents[9];
                std::string rgbFramePath = lineComponents[11];

                float w = std::stof(lineComponents[7]);
                float x = std::stof(lineComponents[4]);
                float y = std::stof(lineComponents[5]);
                float z = std::stof(lineComponents[6]);
                Eigen::Matrix3f rotation = Eigen::Quaternionf(w, x, y, z).toRotationMatrix();

                float tX = std::stof(lineComponents[1]);
                float tY = std::stof(lineComponents[2]);
                float tZ = std::stof(lineComponents[3]);
                Eigen::Vector3f translation = Eigen::Vector3f(tX, tY, tZ);

                Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
                pose.block<3, 3>(0, 0) = rotation;
                pose.block<3, 1>(0, 3) = translation;

                Frame frame = {
                    lineCounter, // lineId
                    lineComponents[0], // id
                    pose, // pose
                    lineComponents[8], // depthId
                    depthFramePath, // depthFramePath
                    lineComponents[10], // rgbId
                    rgbFramePath
                };
                
                m_frames.push_back(frame);

                lineCounter++;
            }

            std::cout << "DataLoader >> Successfully Loaded " << m_frames.size() << " frames" << std::endl;
        }

        ~DataLoader() {
            m_dataDirPath.clear();   
            m_frames.clear();
        }

        Frame getFrameByIndex(int index) {
            return m_frames[index];
        }

        int getDataSetSize() {
            return m_frames.size();
        }

    private:
        std::string m_dataDirPath;
        std::vector<Frame> m_frames;
       
};
#endif