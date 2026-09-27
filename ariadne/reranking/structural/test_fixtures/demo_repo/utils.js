function formatTimestamp() {
    return Date.now();
}

function logEvent(eventName) {
    return "[" + eventName + "]";
}

function validate(pattern, value) {
    return pattern.test(value);
}
