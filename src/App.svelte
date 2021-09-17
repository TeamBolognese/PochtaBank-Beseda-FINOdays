
<!-- [ SCRIPT ] -->
<script>
	// [ IMPORTS: components ]
	import Chat from "./partials/Chat.svelte";
	import Dashboard from "./partials/Dashboard.svelte";
	
	// [ IMPORTS: props ]
	let is_analysing = false;
	let analyticsReceived = false;
	let assessment = 'negative';
	let ratio = {
        positive: 70,
        neutral: 30,
        negative: 0,
    };
	let actions = [];


	// [ METHODS ]
	//@ get > message analytics
	const getAnalysingState = ( e ) => {
		is_analysing = e.detail;
	}
	//@ get > message analytics
	const getAnalytics = ( e ) => {

		// store > analytics results;
		const msg = e.detail;

		// store > all actions
		const action_book = {
			positive: ['поприветствовать, задать вопрос', 'попрощаться, поблагодарить', 'попросить оценить', 'поблагодарить за оценку', 'предложить задать еще вопрос'],
			neutral:  ['neutral action 1', 'neutral action 2'],
			negative: ['быстрый ответ', 'похожие формулировки', 'вызов оператора', 'извиниться', 'заготовленный ответ', 'уточняющий вопрос'],
		};

		// get > data
		assessment = msg.emotion.assessment;
		ratio.positive = msg.emotion.positive;
		ratio.neutral  = msg.emotion.neutral;
		ratio.negative = msg.emotion.negative;
		actions = action_book[assessment];

		// display > analytics in Dashboard
		analyticsReceived = true;

	}

</script>


<!-- [ TEMPLATE: Root ] -->
<main>

	<!-- [ section: chat ] -->
	<Chat 
		on:TransmitAnalytics={ getAnalytics } 
		on:TransmitAnalysingState={ getAnalysingState } 
	/>
	
	<!-- [ section: dashboard ] -->
	<Dashboard 
		{ is_analysing }
		{ analyticsReceived }
		{ assessment }
		{ ratio }
		{ actions }
	/>

</main>