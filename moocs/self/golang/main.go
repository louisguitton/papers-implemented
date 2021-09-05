package main

import (
	"fmt"
	extras "github.com/louisguitton/papers-implemented/moocs/self/golang/extras"
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
}
