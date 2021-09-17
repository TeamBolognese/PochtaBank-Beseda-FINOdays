
// [ IMPORTS: extenstions ]
import { writable } from "svelte/store";


// [ STORE ]
const MessageStore = writable([
	// {
	// 	id: 0,
	// 	message: 'message 0',
	// 	emotion: {
	// 		assessment: 'positive',
	// 		positive: 70,
	// 		neutral:  20,
	// 		negative: 10,
	// 	},
	// 	date: '00:00'
	// },
]);


export default MessageStore;
