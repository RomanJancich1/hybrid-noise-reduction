// Copyright 2011-2021 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include "MyViewer.h"

using namespace nanoflann;
using namespace pmp;


/////////////////////////////////
// Parameters                  //
/////////////////////////////////

const int DIFF_ITERATIONS =12; //Number of iterations for smoothing
const int OPT_ITERATIONS = 7; //Number of iterations for optimization
const float SIGMA = 0.05f; //Weight function parameter
const float BASE_TIMESTEP = 0.4f; //Time step for smoothing
const float THR = 0.00001f; // Parameter ensuring numerical stability
const float MAX_DISPLACEMENT = 0.9f; //Maximum allowed correction displacement in MLS

const float DIFF_STOP_THRESHOLD = 2e-6f; // Diffusion stage stop threshold 
const float OPT_STOP_THRESHOLD = 3e-6f;  // Optimization stage stop threshold

/////////////////////////////////
// Helper Functions            //
/////////////////////////////////

// Extract positions from a mesh
std::vector<Point> extractPoints(const SurfaceMesh& mesh)
{
    std::vector<Point> pts;
    pts.reserve(mesh.n_vertices());
    for (auto v : mesh.vertices())
    {
        pts.push_back(mesh.position(v));
    }
    return pts;
}
// Euclidean norm 
float squaredNorm(const pmp::Point& p)
{
    return p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
}
// Perona - Malik weight function used throughout the algorithm
float computeEdgeStop(float gradNorm)
{
    return std::exp(-(gradNorm * gradNorm) / (SIGMA * SIGMA));
}

/////////////////////////////////
// Noise Functions             //
/////////////////////////////////

// Gaussian noise
static void noise(SurfaceMesh& mesh, double sigma_noise = 0.01)
{
    std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
    std::normal_distribution<> dis(0.0, sigma_noise);

    for (auto v : mesh.vertices())
    {
        auto& point = mesh.position(v);
        point[0] += dis(gen);
        point[1] += dis(gen);
        point[2] += dis(gen);
    }
}

// Speckle noise
static void noise1(SurfaceMesh& mesh, double sigma_noise = 0.0002)
{
    std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
    std::normal_distribution<> dis(0.0, sigma_noise);

    for (auto v : mesh.vertices())
    {
        auto& point = mesh.position(v);
        point[0] *= (1.0 + dis(gen));
        point[1] *= (1.0 + dis(gen));
        point[2] *= (1.0 + dis(gen));
    }
}

// Paul Kellett flicker noise
static void noise2(SurfaceMesh& mesh, double noiseAmplitude = 0.0004)
{
    std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<double> whiteNoise(-1.0, 1.0);

    // State variables for each coordinate channel
    double b0_x = 0, b1_x = 0, b2_x = 0, b3_x = 0, b4_x = 0, b5_x = 0, b6_x = 0;
    double b0_y = 0, b1_y = 0, b2_y = 0, b3_y = 0, b4_y = 0, b5_y = 0, b6_y = 0;
    double b0_z = 0, b1_z = 0, b2_z = 0, b3_z = 0, b4_z = 0, b5_z = 0, b6_z = 0;

    for (auto v : mesh.vertices())
    {
        // --- X coordinate ---
        double white_x = whiteNoise(gen) * noiseAmplitude;
        b0_x = 0.99886 * b0_x + white_x * 0.0555179;
        b1_x = 0.99332 * b1_x + white_x * 0.0750759;
        b2_x = 0.96900 * b2_x + white_x * 0.1538520;
        b3_x = 0.86650 * b3_x + white_x * 0.3104856;
        b4_x = 0.55000 * b4_x + white_x * 0.5329522;
        b5_x = -0.7616 * b5_x - white_x * 0.0168980;
        double pink_x =
            b0_x + b1_x + b2_x + b3_x + b4_x + b5_x + b6_x + white_x * 0.5362;
        b6_x = white_x * 0.115926;

        // --- Y coordinate ---
        double white_y = whiteNoise(gen) * noiseAmplitude;
        b0_y = 0.99886 * b0_y + white_y * 0.0555179;
        b1_y = 0.99332 * b1_y + white_y * 0.0750759;
        b2_y = 0.96900 * b2_y + white_y * 0.1538520;
        b3_y = 0.86650 * b3_y + white_y * 0.3104856;
        b4_y = 0.55000 * b4_y + white_y * 0.5329522;
        b5_y = -0.7616 * b5_y - white_y * 0.0168980;
        double pink_y =
            b0_y + b1_y + b2_y + b3_y + b4_y + b5_y + b6_y + white_y * 0.5362;
        b6_y = white_y * 0.115926;

        // --- Z coordinate ---
        double white_z = whiteNoise(gen) * noiseAmplitude;
        b0_z = 0.99886 * b0_z + white_z * 0.0555179;
        b1_z = 0.99332 * b1_z + white_z * 0.0750759;
        b2_z = 0.96900 * b2_z + white_z * 0.1538520;
        b3_z = 0.86650 * b3_z + white_z * 0.3104856;
        b4_z = 0.55000 * b4_z + white_z * 0.5329522;
        b5_z = -0.7616 * b5_z - white_z * 0.0168980;
        double pink_z =
            b0_z + b1_z + b2_z + b3_z + b4_z + b5_z + b6_z + white_z * 0.5362;
        b6_z = white_z * 0.115926;

        auto& point = mesh.position(v);
        point[0] += pink_x;
        point[1] += pink_y;
        point[2] += pink_z;
    }
}

// Laplacian noise
static void noise3(SurfaceMesh& mesh, double b = 0.001)
{
    std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    auto generateLaplace = [&]() -> double {
        double u = dis(gen);
        return (u < 0.5) ? b * std::log(2 * u) : -b * std::log(2 - 2 * u);
    };

    for (auto v : mesh.vertices())
    {
        auto& point = mesh.position(v);
        point[0] += generateLaplace();
        point[1] += generateLaplace();
        point[2] += generateLaplace();
    }
}

// Partial Gaussian noise (by X coordinate)
static void noiseRightHalfTranslated(SurfaceMesh& mesh,
                                     double sigma_noise = 0.001)
{
    // Compute the minimum and maximum x-values
    float minX = std::numeric_limits<float>::max();
    float maxX = -std::numeric_limits<float>::max();
    for (auto v : mesh.vertices())
    {
        const auto& p = mesh.position(v);
        if (p[0] < minX)
            minX = p[0];
        if (p[0] > maxX)
            maxX = p[0];
    }

    // Center dividing line for the right half
    float centerX = (minX + maxX) / 2.0f;
    // Parameter '0.4f' decides how much will noise cover the mesh
    float offset = 0.4f * (centerX - minX);
    float threshold = centerX - offset;

    std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
    std::normal_distribution<> dis(0.0, sigma_noise);

    // Apply noise to all vertices whose x position is greater than the threshold
    for (auto v : mesh.vertices())
    {
        auto& point = mesh.position(v);
        if (point[0] > threshold)
        {
            point[0] += dis(gen);
            point[1] += dis(gen);
            point[2] += dis(gen);
        }
    }
}

/////////////////////////////////
// Hybrid noise reduction      //
/////////////////////////////////

// Hybrid noise reduction - smoothing

void Diffusion(SurfaceMesh& mesh)
{
    auto points = mesh.vertex_property<Point>("v:point");
    auto newPoints = mesh.add_vertex_property<Point>("v:new_point");

    vertex_normals(mesh);

    for (auto v : mesh.vertices())
    {
        Point grad(0.0f);
        float totalWeight = 0.0f;

        // Compute weighted differences with neighbors
        for (auto vv : mesh.vertices(v))
        {
            Point diff = points[vv] - points[v];
            float distance = std::sqrt(squaredNorm(diff));

            if (distance > THR)
            {
                float weight = computeEdgeStop(distance);
                grad += weight * diff;
                totalWeight += weight;
            }
        }

        // Update by adaptive time step
        if (totalWeight > THR)
        {
            grad /= totalWeight;
            float gradNorm = std::sqrt(squaredNorm(grad));
            float adaptiveStep = BASE_TIMESTEP * (gradNorm / (gradNorm + THR));
            newPoints[v] = points[v] + adaptiveStep * grad;
        }
        else
        {
            newPoints[v] = points[v];
        }
    }

    //updated positions
    for (auto v : mesh.vertices())
    {
        points[v] = newPoints[v];
    }

    mesh.remove_vertex_property(newPoints);
    vertex_normals(mesh);
}

//Hybrid noise reduction - optimization 

// Quadrics correction 
std::vector<Point> QuadricsCorrection(SurfaceMesh& mesh)
{
    auto points = mesh.vertex_property<Point>("v:point");
    auto normals = mesh.vertex_property<Normal>("v:normal");
    vertex_normals(mesh);

    std::vector<Point> corrections(mesh.n_vertices(), Point(0, 0, 0));

    for (auto v : mesh.vertices())
    {
        float Q[3][3] = {{0}};
        float b[3] = {0};

        // Approximation of local geometry
        for (auto vv : mesh.vertices(v))
        {
            auto diff = points[vv] - points[v];
            float weight =
                std::exp(-squaredNorm(diff) / (2.0f * SIGMA * SIGMA));

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                    Q[i][j] += weight * diff[i] * diff[j];
                b[i] += weight * diff[i];
            }
        }

        auto n = normals[v];
        // Penalize displacement in normal direction
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Q[i][j] += 0.1f * n[i] * n[j];

        // Compuation of the correction
        float correction[3] = {0};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                correction[i] += Q[i][j] * b[j];

        // Final projection of quadrics correction
        float dotProduct =
            correction[0] * n[0] + correction[1] * n[1] + correction[2] * n[2];
        for (int i = 0; i < 3; ++i)
            correction[i] -= dotProduct * n[i];

        corrections[v.idx()] =
            Point(correction[0], correction[1], correction[2]);
    }
    return corrections;
}

//Moving least squares correction
std::vector<Point> MLSCorrection(SurfaceMesh& mesh)
{
    auto points = mesh.vertex_property<Point>("v:point");
    std::vector<Point> corrections(mesh.n_vertices(), Point(0, 0, 0));

    for (auto v : mesh.vertices())
    {
        Point centroid(0.0f);
        float totalWeight = 0.0f;

        // Compute weighted centroid of the neighborhood
        for (auto vv : mesh.vertices(v))
        {
            auto diff = points[vv] - points[v];
            float weight =
                std::exp(-squaredNorm(diff) / (2.0f * SIGMA * SIGMA));
            centroid += weight * points[vv];
            totalWeight += weight;
        }

        if (totalWeight > THR)
            centroid /= totalWeight;

        // Penalized displacement 
        Point displacement = centroid - points[v];
        float length = std::sqrt(squaredNorm(displacement));
        if (length > MAX_DISPLACEMENT)
            displacement *= (MAX_DISPLACEMENT / length);

        corrections[v.idx()] = displacement;
    }
    return corrections;
}

//Final combined correction 
void Correction(SurfaceMesh& mesh, float beta)
{
    auto points = mesh.vertex_property<Point>("v:point");

    auto quadricCorr = QuadricsCorrection(mesh);
    auto mlsCorr = MLSCorrection(mesh);

    for (auto v : mesh.vertices())
        points[v] +=
            beta * quadricCorr[v.idx()] + (1 - beta) * mlsCorr[v.idx()];

    vertex_normals(mesh);
}

/////////////////////////////////
// Algorithm evaluation        //
/////////////////////////////////

//Compute the nearest neighbor to a vertex
struct PointCloud
{
    const std::vector<Point>& pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return pts[idx][dim];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

using KDTree = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, PointCloud>,
                                        PointCloud, 3>;

// Compute diameter of the bounding box of the mesh
float computeBoundingBoxDiameter(const std::vector<Point>& points)
{
    Point minPt(std::numeric_limits<float>::max()),
        maxPt(-std::numeric_limits<float>::max());
    for (const auto& p : points)
    {
        for (int d = 0; d < 3; ++d)
        {
            minPt[d] = std::min(minPt[d], p[d]);
            maxPt[d] = std::max(maxPt[d], p[d]);
        }
    }
    return norm(maxPt - minPt);
}

// Computation of normalized symmetric Chamfer distance 
float chamferDistanceNormed(const std::vector<Point>& pointsA,
                            const std::vector<Point>& pointsB)
{
    PointCloud cloudA{pointsA}, cloudB{pointsB};

    KDTree kdTreeA(3, cloudA, KDTreeSingleIndexAdaptorParams(10));
    KDTree kdTreeB(3, cloudB, KDTreeSingleIndexAdaptorParams(10));

    kdTreeA.buildIndex();
    kdTreeB.buildIndex();

    float totalDistAB = 0.0f, totalDistBA = 0.0f;

    // Direction A -> B
    for (const auto& p : pointsA)
    {
        size_t ret_index;
        float out_dist_sqr;
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        kdTreeB.findNeighbors(resultSet, p.data(),
                              nanoflann::SearchParameters());
        totalDistAB += std::sqrt(out_dist_sqr);
    }

    // Direction B -> A
    for (const auto& p : pointsB)
    {
        size_t ret_index;
        float out_dist_sqr;
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        kdTreeA.findNeighbors(resultSet, p.data(),
                              nanoflann::SearchParameters());
        totalDistBA += std::sqrt(out_dist_sqr);
    }

    float avgDistAB = pointsA.empty() ? 0.0f : totalDistAB / pointsA.size();
    float avgDistBA = pointsB.empty() ? 0.0f : totalDistBA / pointsB.size();

    float chamferDist = 0.5f * (avgDistAB + avgDistBA);

    float normFactor = computeBoundingBoxDiameter(pointsA);
    return (normFactor > 0.0f) ? chamferDist / normFactor : chamferDist;
}

////////////////////////////////////////////
// Traditional noise reduction techniques //
////////////////////////////////////////////

//Laplacian filtering
void laplacianFiltering(SurfaceMesh& mesh, int iterations = 3)
{
    for (int iter = 0; iter < iterations; ++iter)
    {
        std::vector<Point> updatedPositions(mesh.n_vertices(), Point(0.0f));

        for (auto v : mesh.vertices())
        {
            Point average(0.0f);
            int count = 0;
            for (auto nv : mesh.vertices(v))
            {
                if (nv != v)
                {
                    average += mesh.position(nv);
                    count++;
                }
            }
            updatedPositions[v.idx()] =
                (count > 0) ? (average / static_cast<float>(count))
                            : mesh.position(v);
        }

        for (auto v : mesh.vertices())
        {
            mesh.position(v) = updatedPositions[v.idx()];
        }
    }
}

//Bilateral filtering
void bilateral(SurfaceMesh& mesh, int iterations = 1)
{
    for (int iter = 0; iter < iterations; ++iter)
    {
        vertex_normals(mesh);
        auto vnormals = mesh.get_vertex_property<Normal>("v:normal");

        std::vector<Point> updated(mesh.n_vertices());
        int idx = 0;

        for (auto v : mesh.vertices())
        {
            Point pi = mesh.position(v);
            Normal ni = vnormals[v];
            float sigma_c = 0.0f;
            for (auto vv : mesh.vertices(v))
            {
                float d = distance(pi, mesh.position(vv));
                sigma_c = std::max(sigma_c, d);
            }

            std::vector<float> offsets;
            float sumOffset = 0.0f;
            for (auto vv : mesh.vertices(v))
            {
                float offset = std::fabs(dot(mesh.position(vv) - pi, ni));
                offsets.push_back(offset);
                sumOffset += offset;
            }
            float avgOffset =
                offsets.empty() ? 0.0f : sumOffset / offsets.size();

            float sigma_s = 1e-8f; 
            for (auto o : offsets)
                sigma_s += (o - avgOffset) * (o - avgOffset);
            sigma_s = std::sqrt(sigma_s / offsets.size());

            // Compute the bilateral filter update along the normal direction
            float weightedSum = 0.0f;
            float weightSum = 0.0f;
            for (auto vv : mesh.vertices(v))
            {
                Point pj = mesh.position(vv);
                float t = distance(pi, pj);
                float h = dot(pj - pi, ni);

                float wc =
                    std::exp(-0.5f * t * t / (sigma_c * sigma_c + 1e-8f));
                float ws =
                    std::exp(-0.5f * h * h / (sigma_s * sigma_s + 1e-8f));

                weightedSum += wc * ws * h;
                weightSum += wc * ws;
            }

            updated[idx++] =
                (weightSum > 0.0f) ? pi + ni * (weightedSum / weightSum) : pi;
        }
        idx = 0;
        for (auto v : mesh.vertices())
        {
            mesh.position(v) = updated[idx++];
        }
    }
}

/////////////////////////////////
// Main                        //
/////////////////////////////////

int main(int argc, char** argv)
{
    const char* inputFile = "vysoke rozlisenie//retheur.obj";
    const char* noisedFile = "noised_mesh.off";
    const char* diffusionFile = "diffusion_mesh.off";
    const char* optimizedFile = "optimized_mesh.off";

    SurfaceMesh originalMesh, noisedMesh, diffusionMesh, optimizedMesh;

    // Load original mesh
    try
    {
        read(originalMesh, inputFile);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to load mesh from " << inputFile << ": "
                  << e.what() << std::endl;
        return 1;
    }

    // Start from the original mesh, then add noise...
    noisedMesh = originalMesh;
    diffusionMesh = originalMesh;
    optimizedMesh = originalMesh;

    // Gaussian noise
   // noise(noisedMesh);
 
    // Speckle noise
     noise1(noisedMesh);

    //Flicker noise
    //noise2(noisedMesh);

    // Laplacian noise
    //noise3(noisedMesh);

    std::vector<Point> originalPositions;
    for (auto v : originalMesh.vertices())
        originalPositions.push_back(originalMesh.position(v));

    // Partial Gaussian noise
    //noiseRightHalfTranslated(noisedMesh, 0.002);

    
    optimizedMesh = originalMesh;


    diffusionMesh = noisedMesh;

    // Smoothing Stage: choose either our method or traditional methods
    std::vector<Point> prevDiffusionPositions = extractPoints(diffusionMesh);

    int diffusionIter;
    for (diffusionIter = 0; diffusionIter < DIFF_ITERATIONS; ++diffusionIter)
    {
        //laplacianFiltering(diffusionMesh, 3);
        Diffusion(diffusionMesh);
        //bilateral(diffusionMesh);
        float cd_iter = chamferDistanceNormed(prevDiffusionPositions,
                                              extractPoints(diffusionMesh));
        std::cout << "Diffusion: Chamfer distance between iteration "
                  << diffusionIter << " and " << diffusionIter + 1 << ": "
                  << cd_iter << std::endl;
        prevDiffusionPositions = extractPoints(diffusionMesh);
    }

    optimizedMesh = diffusionMesh;
    float beta = 0.6f;

   // Optimization Stage (only use with 'Diffusion')
   // Different methods are in smoothing only for comparison purposes, to use them, set OPT_ITERATIONS = 0
    std::vector<Point> prevOptPositions = extractPoints(optimizedMesh);

    for (int optIter = 0; optIter < OPT_ITERATIONS; ++optIter)
    {
        Correction(optimizedMesh, beta);
        float cd_iter = chamferDistanceNormed(prevOptPositions,
                                              extractPoints(optimizedMesh));
        std::cout << "Optimization: Chamfer distance between iteration "
                  << optIter << " and " << optIter + 1 << ": " << cd_iter
                  << std::endl;

        prevOptPositions = extractPoints(optimizedMesh);

        if (cd_iter < OPT_STOP_THRESHOLD)
        {
            std::cout << "Optimization stage converged after iteration "
                      << optIter + 1 << "." << std::endl;
            break;
        }
    }


    try
    {
        write(originalMesh, inputFile);
        write(noisedMesh, noisedFile);
        write(diffusionMesh, diffusionFile);
        write(optimizedMesh, optimizedFile);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to write mesh files: " << e.what() << std::endl;
        return 1;
    }

    // View of original mesh
     /* {
        MyViewer originalWindow("Original Mesh", 800, 600);
        originalWindow.load_mesh(inputFile);
        originalWindow.run();
    } 
 // View of noised mesh
    {
        MyViewer noisedWindow("Noised Mesh", 800, 600);
        noisedWindow.load_mesh(noisedFile);
        noisedWindow.run();
    }
// View of smoothed mesh
     {
        MyViewer diffusionWindow("Mesh after Diffusion", 800, 600);
        diffusionWindow.load_mesh(diffusionFile);
        diffusionWindow.run();
    } */

// View of final optimized mesh
    {
        MyViewer optimizedWindow("Optimized Mesh", 800, 600);
        optimizedWindow.load_mesh(optimizedFile);
        optimizedWindow.run();
    }

    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}
