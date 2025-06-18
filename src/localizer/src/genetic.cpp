#include <cstdlib>
#include <random>
#include <iostream>
#include <chrono>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"
#include "std_msgs/msg/float32.hpp"
#include <opencv2/imgcodecs.hpp>

#define PI 3.141592f

const cv::String mapFilename = "distance_field.png";

cv::Mat mapImage;

int MAP_WIDTH;
int MAP_HEIGHT;

void runGeneticAlgorithm(int numPoints, std::vector<int16_t> const &inputData, std::vector<int16_t> const &boundsData, int &xOut, int &yOut, float &thetaOut);

class GeneticLocaliser: public rclcpp::Node{
    public:
        GeneticLocaliser() : Node("geneticLocaliser"){
            mapImage = cv::imread(mapFilename, cv::IMREAD_GRAYSCALE);
            MAP_HEIGHT = mapImage.rows;
            MAP_WIDTH = mapImage.cols;
            std::cout<<MAP_HEIGHT<<" "<<MAP_WIDTH<<std::endl;
            
            dataSubscription_ = this->create_subscription<std_msgs::msg::Int16MultiArray>("pixelated_image", 10, std::bind(&GeneticLocaliser::dataCallback, this, std::placeholders::_1));
            boundsSubscription_ = this->create_subscription<std_msgs::msg::Int16MultiArray>("bounds", 10, std::bind(&GeneticLocaliser::boundsCallback, this, std::placeholders::_1));
            localisedSpatialCoordinatePublisher_ = this->create_publisher<std_msgs::msg::Int16MultiArray>("localisedSpatialCoordinates", 10);
            localisedRotationalCoordinatePublisher_ = this->create_publisher<std_msgs::msg::Float32>("localisedRotationalCoordinates", 10);
        }
    private:
        bool dataFlag = false;
        bool boundsFlag = false;
        std::vector<int16_t> inputData;
        std::vector<int16_t> boundsData;
        void dataCallback(const std_msgs::msg::Int16MultiArray & msg) {
            if(msg.data.size() <= 0 || dataFlag) return;
            inputData = msg.data;
            dataFlag = true;
            callGeneticAlgorithm();

        }
        void boundsCallback(const std_msgs::msg::Int16MultiArray & msg) {
            if(msg.data.size() <= 0 || boundsFlag) return;
            boundsData = msg.data;
            boundsFlag = true;
            callGeneticAlgorithm();
        }
        void callGeneticAlgorithm(){
            if(dataFlag && boundsFlag){
                int x, y;
                float theta;

                runGeneticAlgorithm(inputData.size()/2, inputData, boundsData, x, y, theta);

                auto spatialMessage = std_msgs::msg::Int16MultiArray();
                spatialMessage.data.resize(2);
                spatialMessage.data[0] = x;
                spatialMessage.data[1] = y;
                localisedSpatialCoordinatePublisher_->publish(spatialMessage);

                auto rotationalMessage = std_msgs::msg::Float32();
                rotationalMessage.data = theta;
                localisedRotationalCoordinatePublisher_->publish(rotationalMessage);

                dataFlag = false;
                boundsFlag = false;
            }
        }
        rclcpp::Subscription<std_msgs::msg::Int16MultiArray>::SharedPtr dataSubscription_;
        rclcpp::Subscription<std_msgs::msg::Int16MultiArray>::SharedPtr boundsSubscription_;
        rclcpp::Publisher<std_msgs::msg::Int16MultiArray>::SharedPtr localisedSpatialCoordinatePublisher_;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr localisedRotationalCoordinatePublisher_;

};

struct individual{
    float x;
    float y;
    float theta;
};

float CROSSOVER_RATE = 0.7f;
float MUTATION_RATE = 0.2f;

float LINEAR_MUTATION_STD_DEV = 30.0f;
float POLAR_MUTATION_STD_DEV = 0.30f;

const int MAX_GENERATIONS = 100;
const int POPULATION_SIZE = 200;
float EARLY_BREAK_THRESHOLD = 0.95;
int PLATEAU_THRESHOLD = 10;

const bool LOGGING = true;

std::default_random_engine generator;

std::uniform_real_distribution<float> normalized(0.0f, 1.0f);

std::uniform_int_distribution<int> populationSize(0, POPULATION_SIZE-1);

std::normal_distribution<float> linearMutation(0.0f, LINEAR_MUTATION_STD_DEV);
std::normal_distribution<float> polarMutation(0.0f, POLAR_MUTATION_STD_DEV);

void initPopulation(int length, individual population[], std::vector<int16_t> const &boundsData){
    for(int i = 0; i < length; i++){
        population[i].x = boundsData[0] + normalized(generator) * (boundsData[1] - boundsData[0]);
        population[i].y = boundsData[2] + normalized(generator) * (boundsData[3] - boundsData[2]);
        population[i].theta = (float)boundsData[4] + normalized(generator) * (float)(boundsData[5] - boundsData[4]);
    }
}

void selectPopulation(int length, float fitnessScores[],individual currGen[] ,individual nextGen[]){
    for(int i = 0; i < length; i++){
        int id1 = populationSize(generator);
        int id2 = populationSize(generator);
        nextGen[i] = (fitnessScores[id1] > fitnessScores[id2]) ? currGen[id1] : currGen[id2];
    }
}

void crossover(individual& parent1, individual& parent2, individual& child1, individual& child2){
    if(normalized(generator) > CROSSOVER_RATE){
        child1 = parent1;
        child2 = parent2;
        return;
    }

    child1.x = (normalized(generator) > 0.5f)?parent1.x:parent2.x;
    child1.y = (normalized(generator) > 0.5f)?parent1.y:parent2.y;
    child1.theta = (normalized(generator) > 0.5f)?parent1.theta:parent2.theta;

    child2.x = (normalized(generator) > 0.5f)?parent1.x:parent2.x;
    child2.y = (normalized(generator) > 0.5f)?parent1.y:parent2.y;
    child2.theta = (normalized(generator) > 0.5f)?parent1.theta:parent2.theta;
    return; 
}

void mutate(individual &indi, std::vector<int16_t> const &boundsData){
    if(normalized(generator) > MUTATION_RATE) return;
    indi.x += linearMutation(generator);
    indi.x = (indi.x>boundsData[1])? boundsData[1]: indi.x;
    indi.x = (indi.x<boundsData[0])? boundsData[0]: indi.x;

    indi.y += linearMutation(generator);
    indi.y = (indi.y>boundsData[3])? boundsData[3]: indi.y;
    indi.y = (indi.y<boundsData[2])? boundsData[2]: indi.y;

    indi.theta += polarMutation(generator);
    indi.theta = (indi.theta>boundsData[5])? (float)boundsData[5]: indi.theta;
    indi.theta = (indi.theta<boundsData[4])? (float)boundsData[4]: indi.theta;

    return;
}

int getFitness(individual &indi, int numPoints, std::vector<int16_t> const &inputData, const int FIXED_POINT_ONE, const int FIXED_SHIFT){
    int newX, newY, sum = 0;
    int cachedSin = (int)(std::sin(indi.theta)*FIXED_POINT_ONE);
    int cachedCos = (int)(std::cos(indi.theta)*FIXED_POINT_ONE);
    int indiXFixedPoint = indi.x*FIXED_POINT_ONE;
    int indiYFixedPoint = indi.y*FIXED_POINT_ONE;
    for(int i = 0; i < 2*numPoints; i+=2){
        newX = (((inputData[i])*cachedCos) - ((inputData[i+1])*cachedSin) + indiXFixedPoint) >> FIXED_SHIFT;
        newY = (((inputData[i])*cachedSin) + ((inputData[i+1])*cachedCos) + indiYFixedPoint) >> FIXED_SHIFT;
        if(newX < 0 || newX >= MAP_WIDTH || newY < 0 || newY >= MAP_HEIGHT) continue;
        sum += mapImage.at<uchar>(newY, newX);
    }
    
    return sum;
}

void runGeneticAlgorithm(int numPoints, std::vector<int16_t> const &inputData, std::vector<int16_t> const &boundsData, int &xOut, int &yOut, float &thetaOut){

    auto start = std::chrono::high_resolution_clock::now();

    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

    float fitnessScores[POPULATION_SIZE];
    individual population[POPULATION_SIZE];
    initPopulation(POPULATION_SIZE, population, boundsData);

    individual bestIndividual = population[0];
    float bestFitness = -INFINITY;

    float currentBestFitness = -INFINITY;
    int currentBestIndividualId = 0;

    float lastBestFitness = -INFINITY;
    int plateauCount = 0;

    float fitnessNormalizationFactor = 1.0f/(255.0f * (float)numPoints);
    const int FIXED_SHIFT = 16;
    const int FIXED_POINT_ONE = 1 << FIXED_SHIFT;

    for(int genNum = 0; genNum < MAX_GENERATIONS; genNum++){
        currentBestFitness = -INFINITY;
        for(int i = 0; i < POPULATION_SIZE; i++){
            fitnessScores[i] = fitnessNormalizationFactor * (float)getFitness(population[i], numPoints, inputData, FIXED_POINT_ONE, FIXED_SHIFT);
            if(fitnessScores[i] > currentBestFitness){
                currentBestFitness = fitnessScores[i];
                currentBestIndividualId = i;
            } 
        }
        if(currentBestFitness > bestFitness){
            bestFitness = currentBestFitness;
            bestIndividual = population[currentBestIndividualId];
        }

        if(bestFitness > EARLY_BREAK_THRESHOLD){
            if(LOGGING) std::cout << "Early break after " << genNum << " generations" << std::endl;
            break;
        }

        if(bestFitness == lastBestFitness){
            plateauCount++;
            if(plateauCount > PLATEAU_THRESHOLD){
                if(LOGGING) std::cout << "Plateau reached after " << genNum << " generations" << std::endl;
                break;
            }
        }else{
            plateauCount = 0;
            lastBestFitness = bestFitness;
        }

        individual selected[POPULATION_SIZE];
        selectPopulation(POPULATION_SIZE, fitnessScores, population, selected);

        for(int i = 0; i+1 < POPULATION_SIZE; i += 2){
            crossover(selected[i], selected[i+1], population[i], population[i+1]);
            mutate(population[i], boundsData);
            mutate(population[i+1], boundsData);
        }

        population[0] = bestIndividual;
        
    }

    auto stop = std::chrono::high_resolution_clock::now();

    xOut = bestIndividual.x;
    yOut = bestIndividual.y;
    thetaOut = bestIndividual.theta;

    if(LOGGING){
        std::cout << "X = " << bestIndividual.x << " Y = " << bestIndividual.y << " Theta = " << bestIndividual.theta << " Fitness = " << bestFitness << std::endl;
        int timeTaken = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        std::cout << 1000000/timeTaken << " fps " << timeTaken << " μs" <<std::endl;
    }
}

int main(int argc, char *argv[]){

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GeneticLocaliser>());
    rclcpp::shutdown();
   
    return 0;
}