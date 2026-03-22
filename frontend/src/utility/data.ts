// data.ts
export let users = JSON.parse(localStorage.getItem("users") || `[
  { "id": "68f3a03f9cd2215b9bfbdd1d", "name": "kavi", "points": 0 },
  { "id": "2", "name": "Kobi", "points": 0 }
]`);

export const saveUsers = () => {
    localStorage.setItem("users", JSON.stringify(users));
};

export const updateUserPoints = (userId: string, quizScore?: number, weight?: number | undefined) => {
    let weightPoints = 0;
    console.log(`Updating points for user ${userId} with quizScore: ${quizScore} and weight: ${weight}`);
    if (weight != undefined) {
        if (weight >= 100) weightPoints = 10;
        else if (weight >= 50) weightPoints = 5;
        else if (weight > 0) weightPoints = 2;
    }
    if (quizScore) {
        quizScore = quizScore * 10; 
    }

    users = users.map((u: any) => u.id === userId ? { ...u, points: u.points + (quizScore || 0) + weightPoints } : u);

    saveUsers();
};