#ifndef OBJECTFINDER_H
#define OBJECTFINDER_H

#include <vector>
#include <typeinfo>

#include "Config.h"
#include "ObjectInfo.h"
#include "Matrix.h"
#include "STBCommons.h"
#include "myMATH.h"

#include "CircleIdentifier.h"

class ObjectFinder2D
{
public:
    ObjectFinder2D() = default;
    ~ObjectFinder2D() = default;

    // Find 2D objects in the image based on the object configuration
    std::vector<std::unique_ptr<Object2D>>
    findObject2D(Image const& img, ObjectConfig const& obj_cfg);

    // std::vector<std::unique_ptr<Object2D>>
    // createTracersFromRawData(const double* centers_ptr, const double* radii_ptr, size_t num_particles);
    std::vector<std::unique_ptr<Object2D>> 
    createTracersFromRawData(const double* centers_ptr, const double* diameters_ptr, const double* intensities_ptr, size_t num_particles);

private:
    std::vector<std::unique_ptr<Object2D>>
    findTracer2D(Image const& img, TracerConfig const& cfg);

    std::vector<std::unique_ptr<Object2D>>
    findBubble2D(Image const& img, BubbleConfig const& cfg);
};


#endif