function greet(name) {
  formatName(name);
  return logMessage(name);
}

const formatName = (name) => {
  return name.trim();
};

async function loadUser(id) {
  const user = await fetchUser(id);
  return greet(user.name);
}

class UserService {
  saveUser(user) {
    validateUser(user);
    return persistUser(user);
  }
}

const wrapper = () => {
  return loadUser(1);
};