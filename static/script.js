const form = document.getElementById("userForm");

form.addEventListener("submit", function(event){
    event.preventDefault();
    const age = parseInt(document.getElementById("age").value);
    var gender = 0;
    if (document.getElementById("gender").value == "male"){
        gender = 1;
    }
    else{
        gender = 0;
    }
    

    let genresArray = new Array(10).fill(4);

    const checkboxes = document.querySelectorAll('input[name="genre[]"]');

    checkboxes.forEach((checkbox, index) => {
        if (checkbox.checked) {
            genresArray[index] = 10;
        }
    });

    //console.log("Age:", age);
    //console.log("Gender:", gender);
    //console.log("Genres array:", genresArray);

    const data = { age, gender, genres: genresArray };

    fetch("http://127.0.0.1:8000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(res => {
        console.log("Server recommendation:", res);
        alert("Recommended movie: " + res.recommended_movie);
    })
    .catch(err => console.error(err));
});