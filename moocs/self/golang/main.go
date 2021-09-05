package main

import (
	"fmt"
	"log"
	"os"

	extras "github.com/louisguitton/papers-implemented/moocs/self/golang/extras"

	"github.com/go-gota/gota/dataframe"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

type Employee struct {
	name string
	age  int
}

func main() {
	fmt.Println("Hello world")
	fmt.Println(extras.HelloFromExtras())

	myseasons := [...]string{"Printemps", "été", "Automne", "Hiver"}
	fmt.Println(myseasons, len(myseasons))

	for i, s := range myseasons {
		fmt.Println(i, s)
	}

	e := Employee{name: "Louis"}
	e.age = 27

	// Load dataset with a dataframe
	irisCsv, err := os.Open("Iris.csv")
	if err != nil {
		log.Fatal(err)
	}

	df := dataframe.ReadCSV(irisCsv)
	fmt.Println(df)

	// Load dataset in golearn
	rawData, err := base.ParseCSVToInstances("Iris.csv", true)
	if err != nil {
		panic(err)
	}

	fmt.Println(rawData)

	// Initialises a new KNN classifier
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)

	// Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.8)
	cls.Fit(trainData)

	// Calculates the Euclidean distance and returns the most popular label
	predictions, err := cls.Predict(testData)
	if err != nil {
		panic(err)
	}
	fmt.Println(predictions)

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))
}
